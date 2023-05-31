##########################################################################################
# hosts/jwst/__init__.py
##########################################################################################

import astropy.io.fits as pyfits
import numpy as np
import os

import julian
import oops
import hosts.solar as solar

from polymath import Vector3

# A handy constant
RADIANS_PER_ARCSEC = oops.RPD / 3600.

INSTRUMENT_NAME = {
    'MIRI'   : 'MIRI',
    'NIRCAM' : 'NIRCam',
    'NIRISS' : 'NIRISS',
    'NIRSPEC': 'NIRSpec',
}

# Define some important paths and frames
oops.Body.define_solar_system('2022-01-01', '2030-01-01')

# Note that the data quality flags are define here:
# https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html
#
# Bit Value       Name              Description
# 0   1           DO_NOT_USE        Bad pixel. Do not use.
# 1   2           SATURATED         Pixel saturated during exposure
# 2   4           JUMP_DET          Jump detected during exposure
# 3   8           DROPOUT           Data lost in transmission
# 4   16          OUTLIER           Flagged by outlier detection
# 5   32          PERSISTENCE       High persistence
# 6   64          AD_FLOOR          Below A/D floor
# 7   128         UNDERSAMP         Undersampling correction
# 8   256         UNRELIABLE_ERROR  Uncertainty exceeds quoted error
# 9   512         NON_SCIENCE       Pixel not on science portion of detector
# 10  1024        DEAD              Dead pixel
# 11  2048        HOT               Hot pixel
# 12  4096        WARM              Warm pixel
# 13  8192        LOW_QE            Low quantum efficiency
# 14  16384       RC                RC pixel
# 15  32768       TELEGRAPH         Telegraph pixel
# 16  65536       NONLINEAR         Pixel highly nonlinear
# 17  131072      BAD_REF_PIXEL     Reference pixel cannot be used
# 18  262144      NO_FLAT_FIELD     Flat field cannot be measured
# 19  524288      NO_GAIN_VALUE     Gain cannot be measured
# 20  1048576     NO_LIN_CORR       Linearity correction not available
# 21  2097152     NO_SAT_CHECK      Saturation check not available
# 22  4194304     UNRELIABLE_BIAS   Bias variance large
# 23  8388608     UNRELIABLE_DARK   Dark variance large
# 24  16777216    UNRELIABLE_SLOPE  Slope variance large (i.e., noisy pixel)
# 25  33554432    UNRELIABLE_FLAT   Flat variance large
# 26  67108864    OPEN              Open pixel (counts move to adjacent pixels)
# 27  134217728   ADJ_OPEN          Adjacent to open pixel
# 28  268435456   FLUX_ESTIMATED    Pixel flux estimated due to missing/bad data
# 29  536870912   MSA_FAILED_OPEN   Pixel sees light from failed-open shutter
# 30  1073741824  OTHER_BAD_PIXEL   A catch-all flag
# 31  2147483648  REFERENCE_PIXEL   Pixel is a reference pixel

MASK_VALUES = {
    'DO_NOT_USE'       : 1,
    'SATURATED'        : 2,
    'JUMP_DET'         : 4,
    'DROPOUT'          : 8,
    'OUTLIER'          : 16,
    'PERSISTENCE'      : 32,
    'AD_FLOOR'         : 64,
    'UNDERSAMP'        : 128,
    'UNRELIABLE_ERROR' : 256,
    'NON_SCIENCE'      : 512,
    'DEAD'             : 1024,
    'HOT'              : 2048,
    'WARM'             : 4096,
    'LOW_QE'           : 8192,
    'RC'               : 16384,
    'TELEGRAPH'        : 32768,
    'NONLINEAR'        : 65536,
    'BAD_REF_PIXEL'    : 131072,
    'NO_FLAT_FIELD'    : 262144,
    'NO_GAIN_VALUE'    : 524288,
    'NO_LIN_CORR'      : 1048576,
    'NO_SAT_CHECK'     : 2097152,
    'UNRELIABLE_BIAS'  : 4194304,
    'UNRELIABLE_DARK'  : 8388608,
    'UNRELIABLE_SLOPE' : 16777216,
    'UNRELIABLE_FLAT'  : 33554432,
    'OPEN'             : 67108864,
    'ADJ_OPEN'         : 134217728,
    'FLUX_ESTIMATED'   : 268435456,
    'MSA_FAILED_OPEN'  : 536870912,
    'OTHER_BAD_PIXEL'  : 1073741824,
    'REFERENCE_PIXEL'  : 2147483648,
}

##########################################################################################
# Standard instrument methods
##########################################################################################

def from_file(filespec, **options):
    """An Observation object based on a given data file generated by the James
    Webb Space Telescope.

    Inputs:
        filespec        path to the FITS file.

    Options:
        data            True (the default) to include the data arrays in the returned
                        Observation.

        calibration     True (the default) to include calibration subfields in the
                        Observation.

        astrometry      If True, this is equivalent to data=False, calibration=False.

        reference       An optional second JWST Observation. If specified, then this
                        Observation will use a frame defined as an offset from that of the
                        reference.

        navigation      An optional tuple/list/array of two or three rotation angles to
                        apply to the frame, yielding a Navigation frame. Use True to
                        employ a Navigation frame without specifying the angles; this is
                        equivalent to navigation=(0.,0.). If not specified, None, or
                        False, a Navigation frame will not be used.

        offset          An optional tuple or Pair of coordinate offsets (du, dv) in units
                        of pixels to apply to the FITS-derived geometry in order to align
                        with the actual image geometry. This is an alternative to
                        specifying the navigation angles; only one of the inputs "offset"
                        and "navigation" can be specified.

        origin          An optional tuple or Pair of coordinate values (u,v) in units of
                        pixels, which define the location in the FOV where the offset was
                        determined. If not provided, the offset is assumed to apply at the
                        center of the FOV.

        parallel        An optional Observation object defining the parallel observation
                        in which the offset and origin parameters are defined. If
                        specified, those options will be converted from the detector of
                        the parallel observation to the detector of this observation.

        frame_suffix    An optional suffix to apply to the name of the observation's
                        frame; by default, just the file basename is used.

        path_suffix     An optional suffix to apply to the name of JWST's path; by
                        default, just the file basename is used.

        target          If specified, the name of the target body. Otherwise, the target
                        body is inferred from the header.

        fast_fov        If True or unspecified, the WCSFOV uses fast inversions using the
                        inverse WCS parameters. If False, it uses the slow method.

    Instrument-specific methods may support additional options.
    """

    # Open the file
    hdulist = pyfits.open(filespec)

    try:
        # Confirm that the telescope is JWST
        if JWST().telescope_name(hdulist) != 'JWST':
            raise IOError('not a JWST file: ' + filespec)

        return JWST.from_hdulist(hdulist, **options)

    finally:
        hdulist.close()

##########################################################################################
# Class JWST
##########################################################################################

class JWST(object):
    """This class defines functions and properties unique to the James Webb Space
    Telescope.

    Objects of this class are empty; they only exist to support inheritance.
    """

    ############################################
    # General file info
    ############################################

    def filespec(self, hdulist, **options):
        """The full directory path and name of the file."""

        # Found by poking around inside a pyfits object
        return hdulist._file.name

    def basename(self, hdulist, **options):
        """The base name of the file."""

        return os.path.basename(hdulist._file.name)

    def telescope_name(self, hdulist, **options):
        """Telescope name, should be "JWST"."""

        return hdulist[0].header['TELESCOP']

    def instrument_name(self, hdulist, **options):
        """Instrument name, one of "NIRCam", "NIRSpec", "MIRI", or "NIRISS"."""

        capname = hdulist[0].header['INSTRUME']
        return INSTRUMENT_NAME[capname]

    def header_subfields(self, hdulist, **options):
        """Default subfields for all JWST Observations."""

        header0 = hdulist[0].header
        header1 = hdulist[1].header

        subfields = {}
        subfields['this'      ] = self
        subfields['filepath'  ] = self.filespec(hdulist, **options)
        subfields['basename'  ] = os.path.basename(subfields['filepath'])
        subfields['headers'   ] = [header0, header1]
        subfields['telescope' ] = 'JWST'
        subfields['instrument'] = INSTRUMENT_NAME[header0['INSTRUME']]
        subfields['detector'  ] = header0['DETECTOR']
        subfields['filter'    ] = header0['FILTER']
        subfields['subarray'  ] = header0['SUBARRAY']
        subfields['target'    ] = self.target_name(hdulist, **options)
        subfields['dither'    ] = {
            'visit'  : header0['VISIT_ID'],
            'step'   : header0['PATT_NUM'],
            'steps'  : header0['NUMDTHPT'],
            'xoffset': header0['XOFFSET'],
            'yoffset': header0['YOFFSET'],
        }

        if 'data' in options:
            include_data = options['data']
        else:
            include_data = not options.get('astrometry', False)

        if include_data:
            subfields['data'] = hdulist['SCI'].data
            if 'ERROR' in hdulist:
                subfields['error'] = hdulist['ERROR'].data
            if 'DQ' in hdulist:
                subfields['quality'] = hdulist['DQ'].data

        return subfields

    def check_options(self, options):
        """Fill in option defaults."""

        astrometry_mode = options.get('astrometry', False)
        options['data'        ] = options.get('data'        , not astrometry_mode)
        options['calibration' ] = options.get('calibration' , not astrometry_mode)
        options['reference'   ] = options.get('reference'   , None)
        options['target'      ] = options.get('target'      , None)
        options['frame_suffix'] = options.get('frame_suffix', '')
        options['path_suffix' ] = options.get('path_suffix' , '')

        if options.get('navigation', False) and options.get('offset', False):
            raise ValueError('navigation and offset values cannot both be specified')

        return options

    ############################################
    # Cadence support
    ############################################

    def time_limits(self, hdulist, **options):
        """A tuple containing the overall start and end times of the observation as
        seconds TDB."""

        header0 = hdulist[0].header
        start_tai = julian.tai_from_iso(header0['DATE-BEG'])
        stop_tai  = julian.tai_from_iso(header0['DATE-END'])
        start_tdb = julian.tdb_from_tai(start_tai)
        stop_tdb  = julian.tdb_from_tai(stop_tai)

        return (start_tdb, stop_tdb)

    def row_cadence(self, hdulist, **options):
        """Default cadence for JWST, with one time step per image row."""

        tstride = hdulist[0].header['TFRAME']       # seconds between frames
        nrows   = hdulist[1].header['NAXIS2']

        (start_tdb, stop_tdb) = self.time_limits(hdulist, **options)

        cadence = oops.cadence.Metronome(tstart = start_tdb,
                                         tstride = tstride/nrows,
                                         texp = stop_tdb - start_tdb - tstride,
                                         steps = nrows)

        if hdulist[0].header['SLOWAXIS'] < 0:
            cadence = oops.cadence.ReversedCadence(cadence)

        return cadence

    ############################################
    # Path support
    ############################################

    def jwst_path(self, hdulist, **options):
        """A LinearPath using the geometry info in the FITS header."""

        h1 = hdulist[1].header

        eph_time = h1['EPH_TIME']   # MJD time of position and velocity vectors
        jwst_x   = h1['JWST_X  ']   # [km] barycentric JWST X coordinate
        jwst_y   = h1['JWST_Y  ']   # [km] barycentric JWST Y coordinate
        jwst_z   = h1['JWST_Z  ']   # [km] barycentric JWST Z coordinate
        jwst_dx  = h1['JWST_DX ']   # [km/s] barycentric JWST X velocity
        jwst_dy  = h1['JWST_DY ']   # [km/s] barycentric JWST Y velocity
        jwst_dz  = h1['JWST_DZ ']   # [km/s] barycentric JWST Z velocity

        epoch = julian.tdb_from_tai(julian.tai_from_mjd(eph_time))
        pos = Vector3((jwst_x , jwst_y , jwst_z ))
        vel = Vector3((jwst_dx, jwst_dy, jwst_dz))
        path_id = self.basename(hdulist, **options) + options['path_suffix']

        return oops.path.LinearPath(pos = (pos, vel),
                                    epoch = epoch,
                                    origin = oops.path.Path.SSB,
                                    frame = oops.frame.Frame.J2000,
                                    path_id = path_id)

    ############################################
    # Frame support
    ############################################

    def target_name(self, hdulist, **options):
        """Name of the image target."""

        if 'target' in options and options['target']:
            return options['target']

        return hdulist[0].header['TARGNAME']

    def target_body(self, hdulist, **options):
        """The body object defining the image target."""

        return oops.Body.lookup(self.target_name(hdulist, **options))

    def tracker_frame(self, hdulist, fov, path, **options):
        """A TrackerFrame for the observation.

        This frame ensures that the target object stays at the same location on the
        detector for the duration of the observation.
        """

        # fov.cmatrix rotates _apparent_ J2000 coordinates to the camera frame.

        # Remove the aberration present in the FOV's C matrix
        header1 = hdulist[1].header
        ephem_mjd = header1['EPH_TIME']
        epoch = julian.tdb_from_tai(julian.tai_from_mjd(ephem_mjd))

        # Insert apparent vector as actual to reverse the aberration effect
        event = oops.path.Path.as_path(path).event_at_time(epoch).wrt_ssb()
        event.neg_arr_j2000 = oops.Vector3.from_ra_dec_length(fov.ra, fov.dec,
                                                              recursive=False)
        (ra, dec) = event.ra_and_dec(apparent=True)

        # This frame describes the actual fixed pointing at epoch, J2000 to FOV
        cmatrix = oops.frame.Cmatrix.from_ra_dec(ra * oops.DPR,
                                                 dec * oops.DPR,
                                                 -fov.clock * oops.DPR)

        # This frame applies for the duration of the observation
        target_body = self.target_body(hdulist, **options)
        frame_id = self.basename(hdulist, **options) + options['frame_suffix']

        return oops.frame.TrackerFrame(frame = cmatrix,
                                       target = target_body.path,
                                       observer = path,
                                       epoch = epoch,
                                       frame_id = frame_id)

    def offset_frame(self, hdulist, reference, **options):
        """A PosTargFrame for the observation, based on a fixed offset from the frame of
        another observation.
        """

        xpos = hdulist[0].header['XOFFSET'] - reference.headers[0]['XOFFSET']
        ypos = hdulist[0].header['YOFFSET'] - reference.headers[0]['YOFFSET']
        xpos *=  RADIANS_PER_ARCSEC
        ypos *= -RADIANS_PER_ARCSEC
        frame_id = self.basename(hdulist, **options) + options['frame_suffix']

        return oops.frame.PosTargFrame(xpos=xpos, ypos=ypos, reference=reference.frame,
                                       frame_id=frame_id)

    def instrument_frame(self, hdulist, fov, path, **options):
        """Either a TrackerFrame or a PosTargFrame, depending on the presence of
        options "reference", "navigation", "offset", etc.
        """

        # Create the frame
        if options.get('reference', None):
            frame = self.offset_frame(hdulist, **options)
        else:
            frame = self.tracker_frame(hdulist, fov=fov, path=path, **options)

        # If navigation is not required, return this frame
        navigation = options.get('navigation', False)
        offset = options.get('offset', False)
        if (not isinstance(navigation, np.ndarray) and not navigation and
            not isinstance(offset, np.ndarray) and not offset):
                return frame

        # Otherwise, return the frame wrapped inside a Navigation frame
        if isinstance(offset, np.ndarray) or offset:
            origin = options.get('origin', None)
            parallel = options.get('parallel', None)
            if parallel:
                angles = parallel.fov.offset_angles_from_duv(offset, origin=origin)
                angles = parallel.parallel_offset_angles((frame, fov), angles)
            else:
                angles = fov.offset_angles_from_duv(offset, origin=origin)
            angles = (angles[0].vals, angles[1].vals)
        elif isinstance(navigation, (tuple, list, np.ndarray)):
            angles = navigation
        else:
            angles = (0., 0.)

        return oops.frame.Navigation(angles, frame, frame_id=frame.frame_id,
                                                    override=True)

    ############################################
    # Calibration support
    ############################################

    def iof_factor(self, hdulist, path, **options):
        """The factor(s) for converting DN per second to I/F."""

        # Check the calibration
        try:
            bunit = hdulist[1].header['BUNIT']
        except KeyError:
            raise IOError('Calibration factors missing from FITS file ' + self.filespec)

        if bunit != 'MJy/sr':
            raise IOError(f'Unrecognized calibration unit {bunit} in FITS file '
                          + self.filespec)

        # Observation event
        times = self.time_limits(hdulist, **options)
        tdb = (times[0] + times[1]) / 2.
        obs_event = oops.Event(tdb, Vector3.ZERO, path, path.frame)

        # Target position as observed
        target_path = self.target_body(hdulist, **options).path
        (target_event, _) = target_path.photon_to_event(obs_event)

        # Sun position as photon source
        sun_path = oops.Body.lookup('SUN').path
        (_, target_event) = sun_path.photon_to_event(target_event)
        solar_range = target_event.arr.norm().vals / solar.AU

        # Get solar F averaged over the filter bandpass
        bandpass = self.filter_bandpass(hdulist, **options)
        solar_model = options.get('solar_model', 'STIS_Rieke')
        solar_f_in_uJy = solar.bandpass_f(bandpass, model=solar_model, units='uJy',
                                          xunits='um', sun_range=solar_range)

        # Convert MJy per steradian to microJy per pixel per dn
        return 1.e12 / solar_f_in_uJy

    ######################################################################################

    @staticmethod
    def from_hdulist(hdulist, **options):
        """An Observation object based on the HDUlist from a JWST FITS data file, plus
        additional options.
        """

        # Make an instance of the JWST class
        jwst = JWST()

        # Confirm that the telescope is JWST
        if jwst.telescope_name(hdulist) != 'JWST':
            raise IOError('not an JWST file: ' + jwst.filespec(hdulist))

        # Select the instrument
        instrument = jwst.instrument_name(hdulist)
        if instrument == 'NIRCam':
            from hosts.jwst.nircam import NIRCam
            obs = NIRCam.from_hdulist(hdulist, **options)

# Someday...
#         elif instrument == 'NIRSPEC':
#             from oops.hosts.jwst.nirspec import NIRSPEC
#             obs = NIRSpec.from_hdulist(hdulist, **options)
#
#         elif instrument == 'MIRI':
#             from oops.hosts.jwst.miri import MIRI
#             obs = MIRI.from_hdulist(hdulist, **options)
#
#         elif instrument == 'NIRISS':
#             from oops.hosts.jwst.niriss import NIRISS
#             obs = NIRISS.from_hdulist(hdulist, **options)

        else:
            raise IOError('unsupported instrument %s in JWST file: %s'
                          % (instrument, jwst.filespec(hdulist)))

        return obs

##########################################################################################
# UNIT TESTS
##########################################################################################

import unittest

class Test_JWST(unittest.TestCase):

    def runTest(self):
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
##########################################################################################
