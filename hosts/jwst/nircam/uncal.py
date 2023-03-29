################################################################################
# hosts/jwst/nircam/uncal.py: JWST/NIRCam subclass for _uncal files
################################################################################

from __future__ import division

import numpy as np
import astropy.io.fits as pyfits

import julian
import oops

from hosts.jwst.nircam     import NIRCam
from hosts.jwst.nircam.cal import Cal

# A handy constant
ARCSEC_PER_RADIAN = 60. * 60. * 180./np.pi

################################################################################
# Standard class method
################################################################################

def from_file(filespec, **options):
    """A TimedImage object based on a given  "_uncal.fits" file from NIRCam.

    The returned TimedImage has a 4-D data array of shape (integrations, groups,
    rows, samples), where the last two dimensions are spatial. Its timing
    represents the entire set of integrations overall.

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

    # Confirm that the file suffix is "_uncal"
    if not filespec.lower().endswith('_uncal.fits'):
        raise ValueError('not an _uncal file: ' + filespec)

    # Open the file
    hdulist = pyfits.open(filespec)

    try:
        # Make an instance of the NIRCam class
        uncal = Uncal()
        filespec = uncal.filespec(hdulist)

        # Confirm that the telescope is JWST
        if uncal.telescope_name(hdulist) != 'JWST':
            raise IOError('not a JWST file: ' + filespec)

        # Confirm that the instrument is NIRCam
        if uncal.instrument_name(hdulist) != 'NIRCam':
            raise IOError('not a JWST/NIRCam file: ' + filespec)

        return uncal.from_hdulist(hdulist, **options)

    finally:
        hdulist.close()

################################################################################
# Class NIRCam.Uncal
################################################################################

class Uncal(NIRCam):

    @staticmethod
    def from_hdulist(hdulist, **options):

        this = Uncal()
        filespec = this.filespec(hdulist)

        subfields = {}
        group_subfields = {}

        ############################################
        # Interpret input options
        ############################################

        options = this.check_options(options)

        options['cal_file'  ] = options.get('cal_file'  , False)
        options['diffs'     ] = options.get('diffs'     , True)
        options['per_second'] = options.get('per_second', True)
        options['groups'    ] = options.get('groups'    , True)

        cal_file = options.get('cal_file', False)
        if cal_file:
            if not isinstance(cal_file, str):
                cal_file = filespec.replace('_uncal.fits', '_cal.fits')
                options['cal_file'] = cal_file

            cal_hdulist = pyfits.open(cal_file)
            cal_image = Cal.from_file(cal_file, **options)

        ############################################
        # Load data
        ############################################

        if options['data'] or options['calibration']:
            data = hdulist['SCI'].data

            if options['diffs']:
                ngroups = data.shape[1]
                for j in range(ngroups, 0, -1):
                    data[:,j] -= data[:,j-1]

        if options['data']:
            subfields['data'] = data

        # Handle the per_second option below...

        ############################################
        # Read header info
        ############################################

        subfields = nircam.header_subfields(hdulist, **options)

        ############################################
        # Define cadences and group exposure times
        ############################################

        # Define the overall cadence
        cadence = this.row_cadence(hdulist, **options)

        # Infer exposure times and validate
        # From https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-
        #  instrumentation/nircam-detector-overview/nircam-detector-subarrays

        noutputs = header0['NOUTPUTS']      # detector outputs used
        nints    = header0['NINTS'   ]      # integrations in exposure
        ngroups  = header0['NGROUPS' ]      # groups in integration
        nframes  = header0['NFRAMES' ]      # frames per group
        groupgap = header0['GROUPGAP']      # frames dropped between groups
        tsample  = header0['TSAMPLE' ]      # microseconds between samples
        tframe   = header0['TFRAME'  ]      # seconds between frames

        nrows    = header1['NAXIS2']
        ncolumns = header1['NAXIS1']

        shape = (nints, ngroups)

        assert noutputs in (4,1)
        if noutputs == 4:
            tframe_calc = ((ncolumns//4 + 12) * (nrows + 1) + 1) * tsample/1.e6
            tframe_wo_telem = (ncolumns//4 + 12) * nrows * tsample/1.e6
        else:
            tframe_calc = (ncolumns + 12) * (nrows + 2) * tsample/1.e6
            tframe_wo_telem = (ncolumns + 12) * nrows * tsample/1.e6

        correction = tframe / tframe_calc
        assert abs(correction - 1) < 0.1, \
            'Something is fishy about the timing of ' + filespec

        tstride = tframe
        texp = tframe_wo_telem * correction

        # Infer the number of frames and number of readouts by group
        group_frames   = np.empty((ngroups,), dtype='int')
        group_readouts = np.empty((ngroups,), dtype='int')

        group_frames[0] = 1
        group_readouts[0] = 1

        if ngroups > 0:
          if options['diffs']:
            if nframes == 1:
                group_frames[1:] = 1
            else:
                group_frames[1] = nframes
                group_frames[2:] = nframes + groupgap

            group_readouts[1]  = nframes
            group_readouts[2:] = 2 * nframes    # readouts before and after

          else:
            iters = np.arange(1, ngroups)
            if nframes == 1:
                group_frames[1:] = iters * (nframes + groupgap) + nframes
            else:
                group_frames[1:] = iters * (nframes + groupgap) - groupgap

            group_readouts[1:] = nframes

        # Infer the duration, effective exposure time, and start time for each
        # group.
        group_duration = texp + group_frames * tstride
        group_texp = group_duration - tstride - (nframes - 1)/2.
        group_texp[0] = texp    # override; first exposure is always one frame

        # Define the cadence of each group
        if options['groups']:

            # Get the stop time of each group
            table = hdulist['GROUP'].data
            i = table['integration_number'] - 1
            j = table['group_number'] - 1
            end_day = table['end_day']
            end_msec = (table['end_milliseconds'] +
                        table['end_submilliseconds']/1000.)

            stop_tai = np.empty(shape)
            stop_tai[i,j] = julian.tai_from_day_sec(end_day, end_msec/1000.)

            start_tai = stop_tai - group_duration
            start_tdb = julian.tdb_from_tai(start_tai)

            group_cadences = np.empty(shape, dtype='object')
            for i in range(nints):
              for j in range(ngroups):
                cadence = oops.cadence.Metronome(
                                        tstart = start_tdb[i,j],
                                        tstride = tstride/nrows,
                                        texp = group_duration[j] - tstride,
                                        steps = nrows)
                if header0['SLOWAXIS'] < 0:
                    cadence = oops.cadence.ReversedCadence(cadence)

                group_cadences[i,j] = cadence

        ############################################
        # Define the FOV
        ############################################

        if options['cal_file']:
            fov = cal_image.fov
        else:
            fov = oops.fov.WCSFOV(header1, ref_axis='y', fast=True)

        ############################################
        # Handle the standard calibrations
        ############################################

        if options['data'] and options['per_second']:
            data /= group_texp[np.newaxis, np.newaxis]

        if options['calibration']:
            arcsec_sq = fov.uv_area * ARCSEC_PER_RADIAN**2
            if options['per_second']:
                cal0 = oops.calib.FlatCalib(name = 'RAW_DN',
                                            factor = group_texp)
                cal1 = oops.calib.NullCalib(name = 'DN_PER_S')
                cal2 = oops.calib.RawCounts(name = 'DN_PER_S_ARCSEC_SQ',
                                            factor = 1. / arcsec_sq,
                                            fov = fov)
            else:
                cal0 = oops.calib.NullCalib(name = 'RAW_DN')
                cal1 = oops.calib.FlatCalib(name = 'DN_PER_S',
                                            factor = 1. / group_texp)
                cal2 = oops.calib.RawCounts(name = 'DN_PER_S_ARCSEC_SQ',
                                            factor = 1. / group_texp
                                                        / arcsec_sq,
                                            fov = fov)
            subfields['raw_dn'] = cal0
            subfields['dn_per_s'] = cal1
            subfields['dn_per_s_arcsec_sq'] = cal2

            if options['groups']:
                group_calibs = {}
                group_cal0 = np.empty((ngroups,), dtype='object')
                group_cal1 = np.empty((ngroups,), dtype='object')
                group_cal2 = np.empty((ngroups,), dtype='object')

                for j in range(ngroups):
                    if options['per_second']:
                        cal0 = oops.calib.FlatCalib(name = 'RAW_DN',
                                                    factor = group_texp[j])
                        cal1 = oops.calib.NullCalib(name = 'DN_PER_S')
                        cal2 = oops.calib.RawCounts(name = 'DN_PER_S_ARCSEC_SQ',
                                                    factor = 1. / arcsec_sq,
                                                    fov = fov)
                    else:
                        cal0 = oops.calib.NullCalib(name = 'RAW_DN')
                        cal1 = oops.calib.FlatCalib(name = 'DN_PER_S',
                                                    factor = 1. / group_texp[j])
                        cal2 = oops.calib.RawCounts(name = 'DN_PER_S_ARCSEC_SQ',
                                                    factor = 1. / group_texp[j]
                                                                / arcsec_sq,
                                                    fov = fov)
                    group_cal0[j] = cal0
                    group_cal1[j] = cal1
                    group_cal2[j] = cal2

                group_calibs['raw_dn'] = group_cal0
                group_calibs['dn_per_s'] = group_cal1
                group_calibs['dn_per_s_arcsec_sq'] = group_cal2

        ############################################
        # Handle the calibrated file...
        ############################################

        if options['cal_file']:

            subfields['cal'] = cal_image
            subfields['cal_file'] = cal_file

            group_subfields['cal'] = cal_image
            group_subfields['cal_file'] = cal_file

            path = cal_image.path

            ############################################
            # Inherit calibrations from cal_file
            ############################################

            if options['calibration'] or options['data']:

                # Use linear regression to get the calibration scale factor and
                # offset; define the calibrations as a function of group
                cal_data = cal_hdulist['SCI'].data
                cal_mask = (cal_hdulist['DQ'].data == 0) & (cal_data > 0)
                cal_factors = np.empty((ngroups,))
                cal_baselines = np.empty((ngroups,))

                # For each group (across all integrations)
                for j in range(ngroups):
                    x = np.mean(data[:,j,cal_mask], axis=0) # mean dns by group
                    y = cal_data[cal_mask]                  # calibrated values
                    (b, a) = np.polyfit(x, y, 1)            # linear fit
                    diffs = y - (b * x + a)                 # compare...
                    diffs /= np.std(diffs, ddof=2)
                    mask2 = np.abs(diffs) <= 4.             # mask out sigma > 4
                    (b, a) = np.polyfit(x[mask2], y[mask2], 1) # redo linear fit
                    cal_factors[j] = b
                    cal_baselines[j] = -a/b

            if options['calibration']:
                # Pre-scale the raw DNs for the NIRCam.Cal image calibration
                for key, subfield in cal_image.subfields.items():
                    if isinstance(subfield, oops.calib.Calibration):
                        subfields[key] = subfield.prescale(cal_factors,
                                                           cal_baselines)
                        if options['groups']:
                            calib = np.empty((ngroups,), dtype='object')
                            for j in range(ngroups):
                                calib[j] = subfield.prescale(cal_factors[j],
                                                             cal_baselines[j])
                            group_calibs[key] = calib

            ############################################
            # Inherit quality and error arrays
            ############################################

            # NIRCam.Cal error is sqrt(VAR_POISSON + VAR_RNOISE + VAR_FLAT)
            # VAR_POISSON - should scale with the total exposure time
            #               contributing to an image.
            # VAR_RNOISE  - should scale with the number of readouts
            #               contributing to an image.
            # VAR_FLAT    - can be ignored because the NIRCam.Uncal images don't
            #               use a flat field image.

            if options['data']:
                var_poisson = cal_hdulist['VAR_POISSON'].data
                var_rnoise = cal_hdulist['VAR_RNOISE'].data

                error = np.empty(data.shape[1:])
                effexptm = header0['EFFEXPTM']
                all_readouts = nints * min((ngroups - 1) * nframes, 1)
                for j in range(ngroups):
                    texp_ratio = group_texp[j] / effexptm
                    readout_ratio = group_readouts[j] / all_readouts
                    error[j] = (np.sqrt(var_poisson * texp_ratio +
                                        var_rnoise * readout_ratio)
                                / cal_factors[j])

                subfields['error'] = error
                subfields['quality'] = cal_hdulist['DQ'].data
                group_subfields['quality'] = subfields['quality']

        else:
            path = this.jwst_path(hdulist, **options)

        ############################################
        # Return the TimedImage...
        ############################################

        frame = this.instrument_frame(hdulist, **options)

        # Construct an array of TimedImages, one for each individual image
        if options['groups']:
            groups = np.empty(shape, dtype='object')
            for i in range(nints):
                for j in range(ngroups):
                    obs = oops.obs.TimedImage(axes = ('i', 'g', 'v', 'u'),
                                              cadence = group_cadences[i,j],
                                              fov = fov,
                                              path = path,
                                              frame = frame,
                                              **group_subfields)

                    if options['data']:
                        obs.insert_subfield('data', data[i,j])

                    if options['data'] and options['cal_file']:
                        obs.insert_subfield('error', error[j])

                    if options['calibration']:
                        for key, calib in group_calibs.items():
                            obs.insert_subfield(key, calib[j])

                    groups[i,j] = obs

            # Make this array a subfield of the overall observation
            subfields['groups'] = groups

        # Construct the overall observation
        obs = oops.obs.TimedImage(axes = ('i', 'g', 'vt', 'u'),
                                  cadence = cadence,
                                  fov = fov,
                                  path = path,
                                  frame = frame,
                                  **subfields)

        return obs

################################################################################
