##########################################################################################
# hosts/jwst/nircam.py: JWST subclass NIRCam
##########################################################################################

import astropy.io.fits as pyfits
import numpy as np
import oops
import os
import tabulation as tab

from hosts.jwst import JWST

# Not currently used, but might be useful...
READ_PATTERNS = {   # (number averaged, stride)
    'RAPID'   : (1,  1),
    'BRIGHT1' : (1,  2),
    'BRIGHT2' : (2,  2),
    'SHALLOW2': (2,  5),
    'SHALLOW4': (4,  5),
    'MEDIUM2' : (2, 10),
    'MEDIUM8' : (8, 10),
    'DEEP2'   : (2, 20),
    'DEEP8'   : (8, 20),
}

##########################################################################################
# Standard class methods
##########################################################################################

def from_file(filespec, **options):
    """A TimedImage object based on a given JWST/NIRCam file.

    Inputs:
        filespec        path to the FITS file.

    Options:
        data            True (the default) to include the data arrays in the returned
                        observation. If this is an uncalibrated image, the "data" subfield
                        is a 4-D array with shape (integrations, groups, rows, samples),
                        where the last two dimensions are spatial. Otherwise, "data" is a
                        2-D array and the subfields "error" and "quality" are also
                        included.

        calibration     True (the default) to include a calibration subfields. Subfield
                        "i_over_f" is included for calibrated images and for uncalibrated
                        images if cal_file is True. For uncalibrated images, subfields
                        "raw_dn", "dn_per_s" and "dn_per_s_arcsec_sq" are also included.

        astrometry      If True, this is equivalent to data=False, calibration=False.

        reference       An optional second JWST Observation. If specified, then this
                        observation will use a frame defined as an offset from that of the
                        reference.

        navigation      An optional tuple/list/array of two or three rotation angles to
                        apply to the frame, yielding a Navigation frame. Use True to
                        employ a Navigation frame without specifying the angles; this is
                        equivalent to navigation=(0.,0.). If not specified, None, or
                        False, a Navigation frame will not be used.

        offset          An optional pair of coordinate offsets (du, dv) in units of pixels
                        to apply to the FITS-derived geometry in order to align with the
                        actual image geometry. This is an alternative to specifying the
                        navigation angles.

        origin          An optional tuple or Pair of coordinate values (u,v) in units of
                        pixels, which define the location in the FOV where the offset was
                        determined. If not provided, the offset is assumed to apply at the
                        center of the FOV.

        frame_suffix    An optional suffix to apply to the name of the observation's
                        frame; by default, just the file basename is used.

        path_suffix     An optional suffix to apply to the name of JWST's path; by
                        default, just the file basename is used.

        target          If specified, the name of the target body. Otherwise, the target
                        body is inferred from the header.

        fast_fov        If True or unspecified, the WCSFOV uses fast inversions using the
                        inverse WCS parameters. If False, it uses the slow method.

    See help(nircam.uncal.from_file) for the additional options related to _uncal.fits.
    """

    # Open the file
    hdulist = pyfits.open(filespec)

    try:
        # Make an instance of the JWST class
        jwst = JWST()

        # Confirm that the telescope is JWST
        if jwst.telescope_name(hdulist) != 'JWST':
            raise IOError('not a JWST file: ' + filespec)

        # Confirm that the instrument is NIRCam
        if jwst.instrument_name(hdulist) != 'NIRCam':
            raise IOError('not a JWST/NIRCam file: ' + filespec)

        return NIRCam.from_hdulist(hdulist, **options)

    finally:
        hdulist.close()

##########################################################################################
# Class NIRCam
##########################################################################################

class NIRCam(JWST):
    """This class defines functions and properties unique to the NIRCam
    instrument.

    Everything else is inherited from higher levels in the class hierarchy.

    Objects of this class are empty; they only exist to support inheritance.
    """

    def filter_bandpass(self, hdulist, **options):
        """Read the filter file for this detector/filter combination, and return
        the bandpass as a Tabulation.
        """

        header = hdulist[0].header
        file_path = (os.environ['OOPS_RESOURCES'].rstrip('/') + '/JWST/NIRCam/'
                     + '%s_%s_system_throughput.txt'
                     % (header['DETECTOR'].replace('LONG','5'),
                        header['FILTER']))
        array = np.loadtxt(file_path, skiprows=1)
        return tab.Tabulation(array[:,0], array[:,1])

    @staticmethod
    def from_hdulist(hdulist, **options):
        """An TimedImage object based on the HDUlist from a JWST FITS data file, plus
        additional options.

        See from_file help for the additional options.
        """

        nircam = NIRCam()

        basename_lc = nircam.basename(hdulist).lower()
        if basename_lc.endswith('_uncal.fits'):
            from hosts.jwst.nircam.uncal import Uncal
            return Uncal.from_hdulist(hdulist, **options)

        if basename_lc[-9:] not in ('_cal.fits', '_i2d.fits'):
            raise ValueError('unsupported NIRCam file type: ' +
                             nircam.basename(hdulist))

        options = nircam.check_options(options)
        subfields = nircam.header_subfields(hdulist, **options)

        fast_fov = options.get('fast_fov', True)
        fov = oops.fov.WCSFOV(hdulist[1].header, ref_axis='y', fast=fast_fov)
        path = nircam.jwst_path(hdulist, **options)
        frame = nircam.instrument_frame(hdulist, fov=fov, path=path, **options)
        cadence = nircam.row_cadence(hdulist, **options)

        if options['calibration']:
            iof_factor = nircam.iof_factor(hdulist, path, **options)
            if basename_lc.endswith('_cal.fits'):
                cal = oops.calib.Radiance(name='I/F', factor=iof_factor,
                                          fov=fov)
            else:
                cal = oops.calib.FlatCalib(name='I/F', factor=iof_factor)

            subfields['i_over_f'] = cal

        subfields['texp'] = subfields['headers'][0]['EFFEXPTM']
        return oops.obs.TimedImage(axes=('vt','u'), cadence=cadence,
                                   fov=fov, path=path, frame=frame,
                                   **subfields)

##########################################################################################
