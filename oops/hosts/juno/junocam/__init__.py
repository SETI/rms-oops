################################################################################
# oops/inst/juno/junocam/__init__.py
################################################################################

import re
import numpy as np
import julian
import cspyce
import pdsparser

import oops

from filecache import FCPath

from oops.hosts.juno import Juno


RATIONALE_RE = re.compile(r' *INS-61504_DISTORTION_Y = ([\d\.]+)')
################################################################################
# Standard class methods
################################################################################

#===============================================================================
def from_file(filespec, fast_distortion=True,
              return_all_planets=False, snap=False, method='strict', **parameters):
    """A general, static method to return a Pushframe object based on a given
    JUNOCAM image file.

    Inputs:
        filespec            Path to input file.

        fast_distortion     True to use a pre-inverted polynomial;
                            False to use a dynamically solved polynomial;
                            None to use a FlatFOV.

        return_all_planets  Include kernels for all planets not just
                            Jupiter or Saturn.

        snap                True to model the image as a Snapshot rather than as
                            a TimedImage.
        method              Label reading method to be passed to Pds3Label.
    """
    JUNOCAM.initialize()    # Define everything the first time through; use
                            # defaults unless initialize() is called explicitly.

    filespec = FCPath(filespec)

    # Load the PDS label
    label = pdsparser.Pds3Label(filespec, method=method).as_dict()

    # Get composite image metadata
    meta = _Metadata(label)

    # Load the data array as separate framelets, with associated labels
    (framelets, flabels) = _load_data(filespec, label, meta)

    # Load time-dependent kernels
    Juno.load_cks(meta.tstart0, meta.tstop)
    Juno.load_spks(meta.tstart0, meta.tstop)

    # Construct a Pushframe for each framelet
    obs = []
    for i in range(meta.nframelets):
        fmeta = _Metadata(flabels[i])

        if snap:
            item = oops.obs.Snapshot(('v','u'),
                                     fmeta.tstart, fmeta.tdi_texp,
                                     fov = fmeta.fov,
                                     path = 'JUNO',
                                     frame = 'JUNO_JUNOCAM',
                                     instrument = 'JUNOCAM',
                                     filter = fmeta.filter,
                                     data = framelets[i,:,:])

        else:

            tdi_lines = fmeta.fov.uv_shape.vals[0]
            cadence = oops.cadence.TDICadence(tdi_lines, fmeta.tstart,
                                              fmeta.tdi_texp, fmeta.tdi_stages)
            fov = oops.fov.TDIFOV(fmeta.fov, tstop=fmeta.tstop,
                                  tdi_texp=fmeta.tdi_texp, tdi_axis='-u')
            item = oops.obs.TimedImage(('v','ut'),
                                       cadence = cadence,
                                       fov = fov,
                                       path = 'JUNO',
                                       frame = 'JUNO_JUNOCAM',
                                       instrument = 'JUNOCAM',
                                       filter = fmeta.filter,
                                       data = framelets[i,:,:])
### The fov value of fov.meta was wrong here. Each framelet needs its own
### TDIFOV, not the generic fov, and each will have its own unique tstop value:
###     fov = TDIFOV(fmeta.fov, tstop, tdi_texp=fmeta.tdi_texp, tdi_axis='-u')
### where tstop is the end time of that individual framelet. I suspect it is
### close enough  that the backplanes look reasonable, but it should be fixed
### and checked.

#        item.insert_subfield('spice_kernels', \
#                   Juno.used_kernels(item.time, 'junocam', return_all_planets))
        item.insert_subfield('filespec', filespec)
        item.insert_subfield('basename', filespec.name)
        obs.append(item)

    return obs

#===============================================================================
def _load_data(filespec, label, meta):
    """Load the data array from the file and splits into individual framelets.

    Input:
        filespec        Full path to the data file.
        label           Label for composite image.
        meta            Image _Metadata object.

    Return:             (framelets, framelet_labels)
        framelets       A Numpy array containing the individual frames in
                        axis order (line, sample, framelet #).
        framelet_labels List of labels for each framelet.
    """

    # Read data
    bits = label['IMAGE']['SAMPLE_BITS']
    dtype = '>u' + str(int(bits // 8))
    local_path = filespec.retrieve()
    data = np.fromfile(local_path, dtype=dtype).reshape(meta.nlines, meta.nsamples)

    # Split into framelets:
    framelets = np.reshape(data, (meta.nframelets, meta.frlines, meta.nsamples))

    # Add framelet parameters to label
    nf = len(meta.filter)
    framelet_labels = []
    for i in range(meta.nframelets):
        framelet_label = label.copy()
        frn = i // nf
        ifl = i % nf

        framelet_label['FRAMELET'] = {}
        framelet_label['FRAMELET']['FRAME_NUMBER'] = frn
        framelet_label['FRAMELET']['FRAMELET_NUMBER'] = i

        filters = label['FILTER_NAME']
        framelet_label['FRAMELET']['FRAMELET_FILTER'] = filters[ifl]

        framelet_labels.append(framelet_label)

    return (framelets, framelet_labels)


#*******************************************************************************
class _Metadata(object):

    #===========================================================================
    def __init__(self, label):
        """Use the label to assemble the image metadata.

        Input:
            label           The label dictionary.

        Attributes:
            nlines          A Numpy array containing the data in axis order
                            (line, sample).
            nsamples        The time sampling array in (line, sample) axis
                            order, or None if no time backplane is found in
                            the file.
            nframelets

        """

        # image dimensions
        self.nlines = label['IMAGE']['LINES']
        self.nsamples = label['IMAGE']['LINE_SAMPLES']
        self.frlines = 128
        self.nframelets = self.nlines // self.frlines

        # Exposure time
        exposure_ms = label['EXPOSURE_DURATION']
        self.exposure = exposure_ms/1000.

        # Filters
        self.filter = label['FILTER_NAME']

        # Default timing for unprocessed frame
        self.tinter = label['INTERFRAME_DELAY']
        self.tinter0 = self.tinter

        self.tstart = julian.tdb_from_tai(
                        julian.tai_from_iso(label['START_TIME']))
        self.tstart0 = self.tstart
        self.tstop = julian.tdb_from_tai(
                       julian.tai_from_iso(label['STOP_TIME']))

        self.tdi_stages = label['JNO:TDI_STAGES_COUNT']
        self.tdi_texp = self.exposure/self.tdi_stages
        if self.exposure < self.tdi_texp:
            self.tdi_texp = self.exposure

        # target
        self.target = label['TARGET_NAME']

        # Framelet-specific parameters, if applicable
        if 'FRAMELET' in label.keys():
            frn = label['FRAMELET']['FRAME_NUMBER']

            # Filter
            self.filter = label['FRAMELET']['FRAMELET_FILTER']

            # Filter-specific instrument id
            instc = {
                'RED':      '-61503',
                'GREEN':    '-61502',
                'BLUE':     '-61501',
                'METHANE':  '-61504' }
            sinstc = instc[self.filter]

            # Timing
            prefix = 'INS' + sinstc
            delta_var = prefix + '_INTERFRAME_DELTA'
            bias_var = prefix + '_START_TIME_BIAS'

            self.delta = cspyce.gdpool(delta_var, 0)[0]
            self.bias = cspyce.gdpool(bias_var, 0)[0]

            self.tinter = self.tinter0 + self.delta
            self.tstart = self.tstart0 + self.bias + frn*self.tinter

            self.tstop = self.tstart + self.exposure

            # FOV
            k1_var = 'INS' + sinstc + '_DISTORTION_K1'
            k2_var = 'INS' + sinstc + '_DISTORTION_K2'
            cx_var = 'INS' + sinstc + '_DISTORTION_X'
            cy_var = 'INS' + sinstc + '_DISTORTION_Y'
            fo_var = 'INS' + sinstc + '_FOCAL_LENGTH'
            px_var = 'INS' + sinstc + '_PIXEL_SIZE'

            k1 = cspyce.gdpool(k1_var, 0)[0]
            k2 = cspyce.gdpool(k2_var, 0)[0]
            cx = cspyce.gdpool(cx_var, 0)[0]
            cy = cspyce.gdpool(cy_var, 0)[0]
            fo = cspyce.gdpool(fo_var, 0)[0]
            px = cspyce.gdpool(px_var, 0)[0]

            # Check RATIONALE_DESC in label for modification to methane
            # DISTORTION_Y
            if self.filter== 'METHANE':
                cy = self.update_cy(label, cy)

            # Construct FOV
            scale = px/fo
            distortion_coeff = [1,0,k1,0,k2]

            self.fov = oops.fov.BarrelFOV(scale,
                                          (self.nsamples, self.frlines),
                                          coefft_uv_from_xy=distortion_coeff,
                                          uv_los=(cx, cy))

        return

    #===========================================================================
    def update_cy(self, label, cy):
        """Look at label RATIONALE_DESC for a correction to DISTORTION_Y for
        some methane images.

        Input:
            label           The label dictionary.
            cy              Uncorrected cy value.

        Output:
            cy              Corrected cy value.

        """
        match = RATIONALE_RE.match(label['RATIONALE_DESC'])
        if match:
            return float(match.group(1))
        return 0

#*******************************************************************************
class JUNOCAM(object):
    """A instance-free class to hold JUNOCAM instrument parameters."""

    instrument_kernel = None
    fovs = {}
    initialized = False

    #===========================================================================
    @staticmethod
    def initialize(asof=None, **kwargs):
        """
        Initialize key information about the JUNOCAM instrument; fill in key
        information about the WAC and NAC.

        Must be called first. After the first call, later calls to this function
        are ignored.

        Input:
            asof        Only use SPICE kernels that existed before this date;
                        None to ignore.
            kwargs:     Arguments for juno.initialize() and Body.define_solar_system()
        """

        # Quick exit after first call
        if JUNOCAM.initialized: return

        # Initialize Juno
        Juno.initialize(asof=asof, **kwargs)
        Juno.load_instruments(asof=asof)

        # Construct the SpiceFrame
        ignore = oops.frame.SpiceFrame("JUNO_JUNOCAM")

        JUNOCAM.initialized = True

    #===========================================================================
    @staticmethod
    def reset():
        """Reset the internal JUNOCAM parameters.

        Can be useful for debugging.
        """

        JUNOCAM.instrument_kernel = None
        JUNOCAM.fovs = {}
        JUNOCAM.initialized = False

        Juno.reset()

################################################################################
