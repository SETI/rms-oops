################################################################################
# oops/inst/jwst/nircam.py: JWST subclass NIRCam
################################################################################

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

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, **options):
    """A TimedImage object based on a given JWST/NIRCam file.

    Inputs:
        filespec        path to the FITS file.

    Options:
        data            True (the default) to include the data arrays in the
                        returned TimedImage.

        calibration     True (the default) to include a calibration subfield
                        "i_over_f" in the TimedImage.

        astrometry      If True, this is equivalent to data=False,
                        calibration=False.

        reference       An optional second JWST Observation. If specified, then
                        this TimedImage will use a frame defined as an offset
                        from that of the reference.

        target          If specified, the name of the target body. Otherwise,
                        the target body is inferred from the header.

    Additional options for _uncal.fits files:
        cal_file        If True or if this is a file path, this indicates that
                        the uncal image arrays should inherit the geometry and
                        calibration of an associated "_cal.fits" file. Default
                        is False. If provided, the returned TimedImage has a
                        subfield "cal" containing the TimedImage of the
                        calibrated image.

        diffs           True (the default) to replace the data in each group
                        (after the first) in the 4-D data array by a successive
                        difference from the previous group.

        per_second      True (the default) divide all array values by the
                        associated exposure time, yielding units of DN/s. If
                        False, the data arrays contain the raw integer DNs.

        groups          True (the default) to in include a subfield "groups" in
                        the returned object. This is an array of shape
                        (integrations, groups), in which each element is an
                        individual TimedImage describing one individual raw
                        image as a 2-D array.

        calibration     True (the default) to include calibration subfields
                        "raw_dn", "dn_per_s" and "dn_per_s_arcsec_sq", in the
                        Observation. If a cal_file is specified, then the
                        "i_over_f" subfield is also provided.
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

################################################################################
# Class NIRCam
################################################################################

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

        header = hdulist[0].heade
        file_path = (os.environ['OOPS_RESOURCES'].rstrip('/') + '/JWST/NIRCam/'
                     + '%s_%s_system_throughput.txt' % (header['DETECTOR'],
                                                        header['FILTER']))
        array = np.loadtxt(file_path, skiprows=1)
        return tab.Tabulation(array[:,0], array[:,1])

    @staticmethod
    def from_hdulist(hdulist, **options):
        """An TimedImage object based on the HDUlist from a JWST FITS data file,
        plus additional options.
        """

        nircam = NIRCam()

        basename_lc = nircam.basename(hdulist).lower()
        if basename_lc.endswith('_uncal.fits'):
            from oops.hosts.jwst.nircam.uncal import Uncal
            return Uncal.from_hdulst(hdulist, **options)

        if basename_lc[-9:] not in ('_cal.fits', '_i2d.fits'):
            raise ValueError('unsupported NIRCam file type: ' +
                             nircam.basename(hdulist))

        options = nircam.check_options(options)
        subfields = nircam.header_subfields(hdulist, **options)

        fov = oops.fov.WCSFOV(hdulist[1].header, ref_axis='y', fast=True)
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

        return oops.obs.TimedImage(axes=('vt','u'), cadence=cadence,
                                   fov=fov, path=path, frame=frame,
                                   **subfields)

################################################################################
