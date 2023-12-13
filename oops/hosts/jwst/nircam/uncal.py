##########################################################################################
# hosts/jwst/nircam/uncal.py: JWST/NIRCam subclass for _uncal files
##########################################################################################

import numpy as np
import astropy.io.fits as pyfits
import julian
import oops

from oops.hosts.jwst.nircam import NIRCam
from oops.hosts.jwst.nircam import from_file as cal_from_file
from oops.hosts.jwst        import MASK_VALUES

# A handy constant
ARCSEC_PER_RADIAN = 60. * 60. * 180./np.pi

# This seems to be roughly the saturation level in the NIRCam detector
RAW_SATURATION = 55000

DEBUG = False

##########################################################################################
# Standard class method
##########################################################################################

def from_file(filespec, **options):
    """A TimedImage object based on a given "_uncal.fits" file from NIRCam.

    The returned TimedImage has a 4-D data array of shape (integrations, groups, rows,
    samples), where the last two dimensions are spatial. Its timing represents the entire
    set of integrations overall.

    Inputs:
        filespec        file path to the FITS file.

    Options:
        data            True (the default) to include the data arrays in the returned
                        TimedImage.

        calibration     True (the default) to include calibration subfields "raw_dn",
                        "dn_per_s" and "dn_per_s_arcsec_sq", in the Observation. If a
                        cal_file is specified, then the "i_over_f" subfield is also
                        provided.

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

        cal_file        If True or if this is a file path, this indicates that the
                        observation should inherit the geometry and calibration of an
                        associated "_cal.fits" file. Default is False. If provided, the
                        returned observation has a subfield "cal" containing the
                        calibrated observation.

        cal_factor      The scale factor to pre-scale raw values of DN per second before
                        inheriting the calibration objects of the cal_file. If not
                        provided along with cal_file, the value will be derived by linear
                        regression, comparing the raw and calibrated data.

        baseline        The baseline value in raw DN to subtract from any image layers
                        that contain the dark current. This is used for the pre-scaling
                        needed to inherit calibrations from the cal_file. If not provided
                        along with cal_file, the value will be derived by linear
                        regression, comparing the raw and calibrated data.

        diffs           True (the default) to replace the data in each group after the
                        first in the 4-D data array by a successive difference from the
                        previous group.

        per_second      True to divide all array values by the associated exposure time,
                        yielding units of DN/s. If False (the default), the data arrays
                        contain the raw integer DNs.

        groups          True (the default) to in include a subfield "groups" in the
                        returned object. This is an array of shape (integrations, groups),
                        in which each element is an individual TimedImage describing one
                        individual raw image as a 2-D array.
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

##########################################################################################
# Class NIRCam.Uncal
##########################################################################################

class Uncal(NIRCam):

    @staticmethod
    def from_hdulist(hdulist, **options):

        this = Uncal()
        filespec = this.filespec(hdulist)

        ############################################
        # Read header info
        ############################################

        subfields = this.header_subfields(hdulist, **options)

        group_subfields = subfields.copy()  # only used if options['groups'] is True
        group_calibs = {}

        ############################################
        # Interpret input options
        ############################################

        options = this.check_options(options)

        options['cal_file'  ] = options.get('cal_file'  , False)
        options['diffs'     ] = options.get('diffs'     , True)
        options['per_second'] = options.get('per_second', False)
        options['groups'    ] = options.get('groups'    , True)
        options['cal_factor'] = options.get('cal_factor', 0.)
        options['baseline'  ] = options.get('baseline'  , None)

        cal_file = options.get('cal_file', False)
        if cal_file:
            if not isinstance(cal_file, str):
                cal_file = filespec.replace('_uncal.fits', '_cal.fits')
                options['cal_file'] = cal_file

            cal_hdulist = pyfits.open(cal_file)
            cal_image = cal_from_file(cal_file, **options)

        ############################################
        # Load data
        ############################################

        if options['data'] or options['calibration']:
            raw_data = hdulist['SCI'].data
            data = raw_data

            if options['diffs']:
                data = data.astype('int32')
                data[1:] -= data[:-1]

        if options['data']:
            subfields['data'] = data

        # Handle the per_second option below...

        ############################################
        # Define cadences and group exposure times
        ############################################

        # Define the overall cadence
        cadence = this.row_cadence(hdulist, **options)

        # Infer exposure times and validate
        # From https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-
        #  instrumentation/nircam-detector-overview/nircam-detector-subarrays

        header0 = hdulist[0].header
        header1 = hdulist[1].header

        noutputs = header0['NOUTPUTS']      # detector outputs used
        nints    = header0['NINTS'   ]      # integrations in exposure
        ngroups  = header0['NGROUPS' ]      # groups per integration
        nframes  = header0['NFRAMES' ]      # frames averaged per group
        groupgap = header0['GROUPGAP']      # frames dropped between groups
        tsample  = header0['TSAMPLE' ]      # microseconds between samples
        tframe   = header0['TFRAME'  ]      # seconds between frames

        nrows    = header1['NAXIS2']
        ncolumns = header1['NAXIS1']

        shape = (nints, ngroups)

        # Infer the number of frames and number of readouts by group
        # group_frames is the number of frames up to the end of each group.
        # group_readouts is the number of frames averaged for each group.
        group_frames   = np.empty((ngroups,), dtype='int')
        group_readouts = np.empty((ngroups,), dtype='int')

        group_frames[0] = 1
        group_readouts[0] = 1

        if ngroups > 1:
            iters = np.arange(1, ngroups)
            if nframes == 1:
                group_frames[1:] = 1 + iters * (groupgap + 1)
            else:
                group_frames[1:] = iters * (nframes + groupgap) - groupgap

            group_readouts[1:] = nframes

        # Get the stop times from the GROUP table
        table = hdulist['GROUP'].data
        table_is_complete = len(table) == nints * ngroups       # WHY CAN THIS BE FALSE?
        if table_is_complete:
            i = table['integration_number'] - 1     # a 1-D array of indices
            g = table['group_number'] - 1           # another 1-D array
            end_day = table['end_day']
            end_msec = (table['end_milliseconds'] + table['end_submilliseconds']/1000.)
            stop_tai = np.empty(shape)
            stop_tai[i,g] = julian.tai_from_day_sec(end_day, end_msec/1000.)
                # Note i, g, end_day, and end_msec are arrays; this handles all times at
                # once.

            # These times enable us to determine the real-world time intervals ("tstride")
            # between frames; this seems to differ a bit from the value of tframe in the
            # header. I don't know why.
            if ngroups == 1:
                tstride = tframe
            elif ngroups == 2:
                tstride = (np.mean(stop_tai[:,1] - stop_tai[:,0])
                           / (group_frames[1] - group_frames[0]))
            else:
                swapped = np.swapaxes(stop_tai, 0, 1)
                coefficients = np.polyfit(group_frames, swapped, deg=1)
                tstride = np.mean(coefficients[0])

            start_tai = stop_tai - tstride * (group_frames + 1)

        # Sometimes, the GROUP table only contains one line, which is useless. When this
        # occurs, the INT_TIMES table also appears to be useless. We do our best to get
        # the timing based on the info we've got.
        else:
            tai_beg = julian.tai_from_iso(header0['DATE-BEG'])
            tai_end = julian.tai_from_iso(header0['DATE-END'])
            tstride_ints = (tai_end - tai_beg) / nints

            # Start times are at uniform intervals by integration
            start_tai = np.empty(shape)
            i = np.arange(nints)
            start_tai[i,:] = (tai_beg + i * tstride_ints)[:,np.newaxis]

            # Stop times are defined by group_frames within the integration
            tstride = tstride_ints / (group_frames[-1] + 1)
            assert abs(tstride - tframe) < 1.e-4, \
                (f'Inferred tstride ({tstride:.8f}) does not match tframe ({tframe}) in '
                 +  filespec)
            stop_tai = start_tai + (group_frames + 1) * tframe

        # The documentation indicates that the tstride includes a small amount of
        # telemetry time, which has to eat into the exposure time, but presumably only
        # once at the end of each readout. We can use these formulas to estimate the
        # portion of the tstride devoted to telemetry.
        #
        # Formulas from https://jwst-docs.stsci.edu/jwst-near-infrared-camera/-
        # nircam-instrumentation/nircam-detector-overview/nircam-detector-subarrays
        assert noutputs in (4,1)
        if noutputs == 4:
            tstride_calc = ((ncolumns//4 + 12) * (nrows + 1) + 1) * tsample/1.e6
            tstride_wo_telem = (ncolumns//4 + 12) * nrows * tsample/1.e6
        else:
            tstride_calc = (ncolumns + 12) * (nrows + 2) * tsample/1.e6
            tstride_wo_telem = (ncolumns + 12) * nrows * tsample/1.e6

        fraction = (tstride_calc - tstride_wo_telem) / tstride_calc
        deadtime = max(fraction * tstride, 0.)

        # Now we can infer the effective duration of each exposure
        group_texp = tstride * (group_frames - (nframes-1)/2.) - deadtime
        group_texp[0] = tstride - deadtime      # but first exposure is always one frame
        stop_tai -= deadtime

        # Determine successive differences between exposure times
        diff_texp = group_texp.copy()
        diff_texp[1:] -= diff_texp[:-1]

        # Apply differences
        if options['diffs']:
            start_tai[:,1:] = stop_tai[:,:-1] - tstride
            group_texp = diff_texp
            group_frames[1:] -= group_frames[:-1]

        # Convert times to TDB
        start_tdb = julian.tdb_from_tai(start_tai)

        # Define the cadence for each integration and group
        if options['groups']:
            group_cadences = np.empty(shape, dtype='object')
            for i in range(nints):
                for g in range(ngroups):
                    cadence = oops.cadence.Metronome(tstart = start_tdb[i,g],
                                                     tstride = tstride/nrows,
                                                     texp = group_texp[g],
                                                     steps = nrows)
                    if header0['SLOWAXIS'] < 0:
                        cadence = oops.cadence.ReversedCadence(cadence)

                    group_cadences[i,g] = cadence

        ############################################
        # Define the FOV
        ############################################

        if options['cal_file']:
            fov = cal_image.fov
        else:
            fast_fov = options.get('fast_fov', True)
            fov = oops.fov.WCSFOV(header1, ref_axis='y', fast=fast_fov)

        ############################################
        # Handle the standard calibrations
        ############################################

        if options['data'] and options['per_second']:
            data = data.astype('float32') / group_texp[..., np.newaxis, np.newaxis]
            subfields['data'] = data

        if options['data'] and options['calibration']:
            arcsec_sq = fov.uv_area * ARCSEC_PER_RADIAN**2
            if options['per_second']:
                cal0 = oops.calib.FlatCalib(name='RAW_DN', factor=group_texp)
                cal1 = oops.calib.NullCalib(name='DN_PER_S')
                cal2 = oops.calib.RawCounts(name='DN_PER_S_ARCSEC_SQ',
                                            factor=1./arcsec_sq, fov=fov)
            else:
                cal0 = oops.calib.NullCalib(name='RAW_DN')
                cal1 = oops.calib.FlatCalib(name='DN_PER_S', factor=1./group_texp)
                cal2 = oops.calib.RawCounts(name='DN_PER_S_ARCSEC_SQ',
                                            factor=1./(group_texp * arcsec_sq), fov=fov)
            subfields['raw_dn'] = cal0
            subfields['dn_per_s'] = cal1
            subfields['dn_per_s_arcsec_sq'] = cal2

            if options['groups']:
                group_cal0 = np.empty((ngroups,), dtype='object')
                group_cal1 = np.empty((ngroups,), dtype='object')
                group_cal2 = np.empty((ngroups,), dtype='object')

                for g in range(ngroups):
                    if options['per_second']:
                        cal0 = oops.calib.FlatCalib(name='RAW_DN', factor=group_texp[g])
                        cal1 = oops.calib.NullCalib(name='DN_PER_S')
                        cal2 = oops.calib.RawCounts(name='DN_PER_S_ARCSEC_SQ',
                                                    factor=1./arcsec_sq, fov=fov)
                    else:
                        cal0 = oops.calib.NullCalib(name='RAW_DN')
                        cal1 = oops.calib.FlatCalib(name='DN_PER_S',
                                                    factor=1./group_texp[g])
                        cal2 = oops.calib.RawCounts(name = 'DN_PER_S_ARCSEC_SQ',
                                                    factor=1./(group_texp[g] * arcsec_sq),
                                                    fov=fov)
                    group_cal0[g] = cal0
                    group_cal1[g] = cal1
                    group_cal2[g] = cal2

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

                if options['cal_factor'] and options['baseline'] is not None:
                    a = options['cal_factor']
                    b = options['baseline'] / diff_texp[0]
                else:
                    (a, b) = Uncal.fit_to_calibrated(hdulist, cal_hdulist,
                                                     diff_texp=diff_texp,
                                                     cal_factor=options['cal_factor'])
                    if options['baseline'] is not None:
                        b = options['baseline'] / diff_texp[0]

                # a and b convert raw DN PER SECOND to calibrated values
                cal_factor = a

                # The baseline value we save is the value to subtract from the raw DNs,
                # not scaled by time.
                if options['baseline'] is None:
                    baseline = b * diff_texp[0]
                else:
                    baseline = options['baseline']

                # These values are organized by group and applicable to the data arrays
                # as returned.
                if options['per_second']:
                    cal_factors = np.broadcast_to(a, (ngroups,))
                    cal_baselines = baseline / group_texp
                else:
                    cal_factors = a / group_texp
                    cal_baselines = np.broadcast_to(baseline, (ngroups,)).copy()
                        # copy is needed so we can modify cal_baselines below if necessary

                if options['diffs']:
                    cal_baselines[1:] = 0.

            if options['calibration']:

                # Pre-scale the raw DNs for the image calibration
                for key, subfield in cal_image.subfields.items():
                    if key in ('raw_dn', 'dn_per_s', 'dn_per_s_arcsec_sq'):
                        continue
                    if not isinstance(subfield, oops.calib.Calibration):
                        continue

                    subfields[key] = subfield.prescale(cal_factors, cal_baselines)
                    if options['groups']:
                        calib = np.empty((ngroups,), dtype='object')
                        for g in range(ngroups):
                            calib[g] = subfield.prescale(cal_factors[g], cal_baselines[g])
                        group_calibs[key] = calib

            ############################################
            # Inherit quality and error arrays
            ############################################

            # NIRCam calibrated error is sqrt(VAR_POISSON + VAR_RNOISE + VAR_FLAT)
            # VAR_POISSON - scales with the total exposure time contributing to an image.
            # VAR_RNOISE  - scales with the number of readouts contributing to an image.
            # VAR_FLAT    - can be ignored because the NIRCam.Uncal images don't use a
            #               flat field image.

            if options['calibration']:
                var_poisson = cal_hdulist['VAR_POISSON'].data
                var_rnoise = cal_hdulist['VAR_RNOISE'].data

                # Determine the total number of readouts of the detector across all
                # integrations and groups.
                if nframes == 1:
                    all_readouts = nints * ngroups
                else:
                    all_readouts = nints * (ngroups - 1) * nframes

                error = np.empty(data.shape[1:])
                effexptm = header0['EFFEXPTM']  # effective exposure time overall
                for g in range(ngroups):
                    texp_ratio = group_texp[g] / effexptm
                    readout_ratio = group_readouts[g] / all_readouts
                    error[g] = np.sqrt(var_poisson * texp_ratio +
                                       var_rnoise * readout_ratio) / cal_factors[g]

                quality = cal_hdulist['DQ'].data
                quality = np.broadcast_to(quality, data.shape[1:]).copy()

                saturated = np.mean(raw_data, axis=0) > RAW_SATURATION
                quality[saturated] |= MASK_VALUES['SATURATED']

                subfields['error'] = np.broadcast_to(error, data.shape)
                subfields['quality'] =  np.broadcast_to(quality, data.shape)

        else:
            path = this.jwst_path(hdulist, **options)

        ############################################
        # Return the TimedImage...
        ############################################

        if cal_file:
            frame = cal_image.frame
        else:
            frame = this.instrument_frame(hdulist, fov=fov, path=path, **options)

        # Construct an array of TimedImages, one for each individual image
        if options['groups']:
            groups = np.empty(shape, dtype='object')
            for i in range(nints):
                for g in range(ngroups):
                    obs = oops.obs.TimedImage(axes = ('i', 'g', 'vt', 'u'),
                                              cadence = group_cadences[i,g],
                                              fov = fov,
                                              path = path,
                                              frame = frame,
                                              **group_subfields)

                    if options['data']:
                        obs.insert_subfield('data', data[i,g])

                    if options['cal_file'] and options['calibration']:
                        obs.insert_subfield('error', error[g])
                        obs.insert_subfield('quality', quality[g])
                        obs.insert_subfield('cal_factor', cal_factors[g])
                        obs.insert_subfield('baseline', cal_baselines[g])

                    if options['calibration']:
                        for key, calib in group_calibs.items():
                            obs.insert_subfield(key, calib[g])

                    obs.insert_subfield('texp', group_texp[g])
                    obs.insert_subfield('frames', group_frames[g])
                    obs.insert_subfield('readouts', group_readouts[g])

                    groups[i,g] = obs

            # Make this array a subfield of the overall observation
            subfields['groups'] = groups

        # Construct the overall observation
        subfields['texp'] = group_texp
        subfields['frames'] = group_frames
        subfields['readouts'] = group_readouts
        subfields['shape'] = shape + (ncolumns, nrows)

        if options['cal_file'] and options['calibration']:
            subfields['cal_factor'] = cal_factor
            subfields['baseline'] = baseline

        obs = oops.obs.TimedImage(axes = ('i', 'g', 'vt', 'u'),
                                  cadence = cadence,
                                  fov = fov,
                                  path = path,
                                  frame = frame,
                                  **subfields)
        return obs

    @staticmethod
    def fit_to_calibrated(raw_hdulist, cal_hdulist, diff_texp, cal_factor=0.):
        """Solve for the best-fit slope and group-zero offset to match the calibrated
        image to the raw images in units of DN/second.

        If cal_factor is provided, then it is returned as the slope and only the baseline
        is solved.

        Return value is (slope, baseline), such that
            slope * (raw_DN_per_second - baseline) ~ calibrated_DN
        for images that contain the dark current, or
            slope * raw_DN_per_second ~ calibrated_DN
        for those that do not.
        """

        # Get the raw data; define the initial antimask
        raw_data = raw_hdulist['SCI'].data.astype('float32')
        antimask = (raw_data != 0.) & (raw_data <= RAW_SATURATION)

        # Average over integrations for each group index
        raw_data = np.median(raw_data, axis=0)
        antimask = np.logical_or.reduce(antimask, axis=0)

        # Define successive differences; convert to DN per second
        raw_data[1:] -= raw_data[:-1]
        raw_data /= diff_texp[..., np.newaxis, np.newaxis]

        # Each layer of the antimask is False where any layer beneath it is False
        for i in range(0, len(antimask)-1):
            antimask[i+1] &= antimask[i]

        # Get cal_data; broadcast to match the shape of raw_data
        cal_data = cal_hdulist['SCI'].data.copy()
        cal_data[np.isnan(cal_data)] = 0.
        cal_data = np.broadcast_to(cal_data, raw_data.shape)

        # Incorporate the calibrated mask
        antimask &= (cal_hdulist['DQ'].data == 0) & (cal_data > 0)

        # We will determine coefficients (a,b) in:
        #   Y = a * X1 + b * X0
        # where
        #   X1 = raw_data;
        #   X0 = one for group 0, zero elsewhere.
        # With this definition of X0, we are requiring the constant term in the
        # fit to apply only to group 0.
        #
        # The goal is to solve for (a,b) such that the residuals between Y and
        # cal_data are minimized.

        # If the cal_factor is provided, just solve for b
        if cal_factor:
            antimask = antimask[0]
            unmasked = np.sum(antimask)
            y = cal_data[0]
            x = raw_data[0]
            a = cal_factor

            for count in range(10):
                diff = y - a * x
                selected = diff[antimask]
                b = np.mean(selected)
                stdev = np.std(selected)
                antimask &= np.abs(diff - b) < 4. * stdev

                new_unmasked = np.sum(antimask)
                delta = unmasked - new_unmasked
                unmasked = new_unmasked

                if DEBUG:
                    print(count, unmasked, delta, stdev, a, b)

                if delta <= 10:
                    break

            # Y = a * X1 + b * X0 -> Y = a * (X1 + b/a)
            return (a, -b/a)

        # We can rewrite this as a matrix equation:
        #   Y = X A
        # where
        #   X is a matrix of two columns [X0 X1] and A is the matrix [a b].
        #
        # The solution is:
        #   A = (XT W X)^-1 XT W Y
        #
        # See https://en.wikipedia.org/wiki/Weighted_least_squares

        # We can pre-calculate a bunch of stuff
        x = raw_data
        y = cal_data
        w = y                       # larger amplitudes get higher weight

        xcol0 = x
        xcol1 = np.zeros(raw_data.shape, dtype='int32')
        xcol1[0] = 1

        w_xcol0_xcol0 = w * xcol0**2
        w_xcol0_xcol1 = w * xcol0 * xcol1
        w_xcol1_xcol1 = w * xcol1   # xcol1 is zero or one so no need to square this

        w_xcol0_y = w * xcol0 * y
        w_xcol1_y = w * xcol1 * y

        # Iterate while excluding large residuals
        unmasked = np.sum(antimask)
        for count in range(10):

            # Solve A = (XT X)^-1 XT Y
            xt_x_00 = np.sum(antimask * w_xcol0_xcol0)
            xt_x_01 = np.sum(antimask * w_xcol0_xcol1)
            xt_x_11 = np.sum(antimask * w_xcol1_xcol1)

            det = xt_x_00 * xt_x_11 - xt_x_01**2
            det_inv = 1. / det
            xt_x_inv_00 =  xt_x_11 * det_inv
            xt_x_inv_01 = -xt_x_01 * det_inv
            xt_x_inv_11 =  xt_x_00 * det_inv

            xt_y_0 = np.sum(antimask * w_xcol0_y)
            xt_y_1 = np.sum(antimask * w_xcol1_y)

            a = xt_x_inv_00 * xt_y_0 + xt_x_inv_01 * xt_y_1
            b = xt_x_inv_01 * xt_y_0 + xt_x_inv_11 * xt_y_1

            model = a * xcol0 + b * xcol1
            resid = y - model
            stdev = np.std(resid[antimask])     # ignore first group here
            antimask &= np.abs(resid) < 4. * stdev

            new_unmasked = np.sum(antimask)
            delta = unmasked - new_unmasked
            unmasked = new_unmasked

            if DEBUG:
                print(count, unmasked, delta, stdev, a, b)

            if delta <= 10:
                break

        return (a, -b/a)

##########################################################################################
