##########################################################################################
# oops/hosts/cassini/uvis.py
##########################################################################################

import numpy as np
import numbers
import os

import julian
import oops
import pdsparser

from oops.hosts.cassini import Cassini
from oops.hosts         import pds3

from filecache import FCPath

DEBUG = False       # True to assert that the data array must have null
                    # values outside the active windows

##########################################################################################
# Standard class methods
##########################################################################################

def from_file(filespec, data=True, enclose=False, **parameters):
    """A general, static method to return one or more Observation subclass
    objects based on a label for a given Cassini UVIS file.

    Input:
        filespec        the full path to the PDS label of a UVIS data file.
        data            True to include the data array.
        enclose         True to return a single observation, regardless of how many
                        windows are defined. If multiple windows are used, then the
                        observation (and the optional data array) are are defined by the
                        enclosing limits in line and band, and the binning is assumed to
                        be 1. If False and multiple windows are used, the function returns
                        a tuple of observations rather than a single observation.
    """

    UVIS.initialize()   # Define everything the first time through; use defaults unless
                        # initialize() is called explicitly.

    # Get the label dictionary and data array dimensions
    label = pds3.fast_dict(filespec)

    # Load any needed SPICE kernels
    tstart = julian.tdb_from_tai(julian.tai_from_iso(label['START_TIME']))
    tstop  = julian.tdb_from_tai(julian.tai_from_iso(label['STOP_TIME']))
    Cassini.load_cks( tstart, tstop)
    Cassini.load_spks(tstart, tstop)

    # Figure out the PDS object class and return the observation(s)
    if 'QUBE' in label:
        return get_qube(filespec, tstart, label, data, enclose)
    elif 'TIME_SERIES' in label:
        return get_time_series(filespec, tstart, label, data)
    else:
        return get_spectrum(filespec, tstart, label, data)

#=========================================================================================
def get_qube(filespec, tstart, label, data, enclose):
    """The observation object given that it is a QUBE."""

    global DEBUG

    # Determine the detector and mode
    detector = label['PRODUCT_ID'][:3]
    assert detector in ('EUV', 'FUV')

    resolution = label['SLIT_STATE']

    # Define the instrument frame
    frame_id = UVIS.frame_ids[detector]

    # Get array shape
    info = label['QUBE']
    (bands,lines,samples) = info['CORE_ITEMS']
    assert lines in (1,64), f'invalid lines = {lines}'

    if lines == 1:
        shape = (samples, bands)
    else:
        shape = (lines, samples, bands)

    # Define the cadence
    texp = label['INTEGRATION_DURATION']
    cadence = oops.cadence.Metronome(tstart, texp, texp, samples)

    # Define the full FOV
    fov = UVIS.fovs[(detector, label['SLIT_STATE'], lines)]

    # Load the data array if necessary
    info = label['QUBE']
    assert info['CORE_ITEM_TYPE'] == 'MSB_UNSIGNED_INTEGER'
    assert info['CORE_ITEM_BYTES'] == 2
    assert info['SUFFIX_ITEMS'] == [0,0,0]

    if data:
        array_null = 65535                  # Incorrectly -1 in many labels
        array = load_data(filespec, label['^QUBE'], '>u2')

        # Re-shape into something sensible
        # Note that the axis order in the label is first-index-fastest
        if lines > 1:
            array = array.reshape((samples,lines,bands))
            array = array.swapaxes(0,1)
        else:
            array = array.reshape(shape)
    else:
        array = None

    # Identify the window(s) used
    # Note that these are either integers or lists of integers
    line0 = info['UL_CORNER_LINE']
    line1 = info['LR_CORNER_LINE']
    line_bin = info['LINE_BIN']

    band0 = info['UL_CORNER_BAND']
    band1 = info['LR_CORNER_BAND']
    band_bin = info['BAND_BIN']

    # Check the outer periphery of the data array in DEBUG mode
    if DEBUG and data:
        assert np.all(array[:min(line0),    ...] == array_null)
        assert np.all(array[ max(line1)+1:, ...] == array_null)

        assert np.all(array[..., :min(band0)   ] == array_null)
        assert np.all(array[...,  max(band1)+1:] == array_null)

    # One window
    if isinstance(line0, numbers.Integral):
        return get_one_qube(label, detector, resolution,
                            fov, cadence, frame_id,
                            shape, array, samples,
                            lines, line0, line1+1, line_bin,
                            bands, band0, band1+1, band_bin,
                            rebin=True)

    # Multiple windows combined into one enclosure
    elif enclose:
        line0 = min(line0)
        line1 = max(line1)
        line_bin = min(line_bin)
        band0 = min(band0)
        band1 = max(band1)
        band_bin = min(band_bin)
        return get_one_qube(label, detector, resolution,
                            fov, cadence, frame_id,
                            shape, array, samples,
                            lines, line0, line1+1, line_bin,
                            bands, band0, band1+1, band_bin,
                            rebin=False)

    # Separate windows
    else:
        obslist = []
        for w in len(line0):
            obs = get_one_qube(label, detector, resolution,
                               fov, cadence, frame_id,
                               shape, array, samples,
                               lines, line0[w], line1[w]+1, line_bin[w],
                               bands, band0[w], band1[w]+1, band_bin[w],
                               rebin=True)
            obslist.append(obs)

        return tuple(obslist)

#=========================================================================================
def get_one_qube(label, detector, resolution,
                 fov, cadence, frame_id,
                 shape, array, samples,
                 lines, line0, line1, line_bin,
                 bands, band0, band1, band_bin,
                 rebin):
    """A single Observation object for the identified window of the UVIS qube."""

    global DEBUG

    # Trim the lines
    dline = line1 - line0
    if (line0,line1) != (0,lines):
        fov = oops.fov.SliceFOV(fov, (0,line0), (1,dline))
        shape = (dline,) + shape[1:]

        if array is not None:
            array = array[line0:line1, :]

    # Trim the bands
    dband = band1 - band0
    if (band0,band1) != (0,bands):
        shape = shape[:-1] + (dband,)

        if array is not None:
            array = array[..., band0:band1]

    # Bin the lines
    if rebin and line_bin > 1:
        assert dline % line_bin == 0
        fov = oops.fov.SubsampledFOV(fov, (1,line_bin))
        dline_binned = dline // line_bin
        shape = (dline_binned,) + shape[1:]

        if array is not None:
            if DEBUG:
                assert np.all(array[dline_binned:, ...] == array_null)

            array = array[:dline_binned]

    # Bin the bands
    if rebin and band_bin > 1:
        if DEBUG:
            assert dband % band_bin == 0    # seen to fail occasionally

        dband_binned = dband // band_bin
        shape = shape[:-1] + (dband_binned,)

        if array is not None:
            if DEBUG:
                assert np.all(array[..., dband_binned:] == array_null)

            array = array[..., :dband_binned]

    # Create the Observation
    if lines == 1:
        obs = oops.obs.Pixel(('t','b'), cadence, fov, 'CASSINI', frame_id)
    else:
        obs = oops.obs.TimedImage(('v','ut','b'), cadence, fov, 'CASSINI', frame_id)

    obs.insert_subfield('dict', label)
    obs.insert_subfield('instrument', 'UVIS')
    obs.insert_subfield('detector', detector)
    obs.insert_subfield('sampling', resolution)
    obs.insert_subfield('product_type', 'QUBE')

    obs.insert_subfield('line_window', (line0,line1))
    obs.insert_subfield('line_bin', line_bin)

    obs.insert_subfield('band_window', (band0,band1))
    obs.insert_subfield('band_bin', band_bin)

    obs.insert_subfield('samples', samples)

    if array is not None:
        obs.insert_subfield('data', array)

    # Update the observation shape
    obs.shape = shape

    return obs

#=========================================================================================
def get_time_series(filespec, tstart, label, data):
    """The observation object given that it is a TIME_SERIES."""

    # Determine the detector
    product_id = label['PRODUCT_ID']
    if product_id.startswith('HSP'):
        detector = 'HSP'
    elif product_id.startswith('HDAC'):
        detector = 'HDAC'
    else:
        raise ValueError('Time series is neither HSP nor HDAC: ' + filespec)

    # Define the instrument frame
    frame_id = UVIS.frame_ids[detector]

    # Get the array shape
    info = label['TIME_SERIES']
    samples = info['ROWS']

    # Define the cadence
    assert info['COLUMNS'] == 1
    assert (info['SAMPLING_PARAMETER_UNIT'] == 'MILLISECOND' or
            info['SAMPLING_PARAMETER_UNIT'] == 'MILLISECONDS')
    texp = info['SAMPLING_PARAMETER_INTERVAL'] * 0.001

    cadence = oops.cadence.Metronome(tstart, texp, texp, samples)

    # Define the observation
    fov = UVIS.fovs[(detector, '', 1)]
    obs = oops.obs.Pixel(('t',), cadence, fov, 'CASSINI', frame_id)

    obs.insert_subfield('dict', label)
    obs.insert_subfield('instrument', 'UVIS')
    obs.insert_subfield('detector', detector)
    obs.insert_subfield('product_type', 'TIME_SERIES')

    obs.insert_subfield('line_window', None)
    obs.insert_subfield('line_bin', None)

    obs.insert_subfield('band_window', None)
    obs.insert_subfield('band_bin', None)

    obs.insert_subfield('samples', samples)

    # Load the data array if necessary
    if data:
        column = info['PHOTOMETER_COUNTS']
        assert column['DATA_TYPE'] == 'MSB_UNSIGNED_INTEGER'
        assert column['BYTES'] == 2

        array = load_data(filespec, label['^TIME_SERIES'], '>u2')
        obs.insert_subfield('data', array)

    # Update the observation shape
    obs.shape = (samples,)

    return obs

#=========================================================================================
def get_spectrum(filespec, tstart, label, data):
    """The observation object given that it is a SPECTRUM."""

    # Determine the detector
    detector = label['PRODUCT_ID'][:3]
    assert detector in ('EUV', 'FUV')

    # Define the instrument frame
    frame_id = UVIS.frame_ids[detector]

    # Get array shape
    info = label['SPECTRUM']
    bands = info['ROWS']

    # Define the cadence (such as it is)
    assert info['COLUMNS'] == 1
    texp = label['INTEGRATION_DURATION']
    cadence = oops.cadence.Metronome(tstart, texp, texp, 1)

    # Define the FOV
    resolution = label['SLIT_STATE']
    fov = UVIS.fovs[(detector, resolution, 64)]

    line0 = info['UL_CORNER_SPATIAL']
    line1 = info['LR_CORNER_SPATIAL'] + 1
    if (line0,line1) != (0,64):
        fov = oops.fov.SliceFOV(fov, (0,line0), (1,line1-line0))

    line_bin = info['BIN_SPATIAL']
    if line_bin != 1:
        fov = oops.fov.SubsampledFOV(fov, (1,line_bin))

    # Define the observation
    obs = oops.obs.Pixel(('b',), cadence, fov, 'CASSINI', frame_id)

    obs.insert_subfield('dict', label)
    obs.insert_subfield('instrument', 'UVIS')
    obs.insert_subfield('detector', detector)
    obs.insert_subfield('sampling', resolution)
    obs.insert_subfield('product_type', 'SPECTRUM')

    obs.insert_subfield('line_window', (line0,line1))
    obs.insert_subfield('line_bin', line_bin)

    obs.insert_subfield('band_window', (info['UL_CORNER_SPECTRAL'],
                                        info['LR_CORNER_SPECTRAL']+1))
    obs.insert_subfield('band_bin', info['BIN_SPECTRAL'])

    obs.insert_subfield('samples', 1)

    # Load the data array if necessary
    if data:
        column = info['SPECTRUM']
        assert column['DATA_TYPE'] == 'MSB_UNSIGNED_INTEGER'
        assert column['BYTES'] == 2

        array = load_data(filespec, label['^SPECTRUM'], '>u2')
        obs.insert_subfield('data', array)

    # Update the observation shape
    obs.shape = (bands,)

    return obs

#=========================================================================================
def load_data(filespec, body, dtype):

    data_filespec = FCPath(filespec).with_name(body)

    try:
        local_path = data_filespec.retrieve()
    except FileNotFoundError:
        data_filespec = data_filespec.with_name(body.lower())
        local_path = data_filespec.retrieve()

    return np.fromfile(local_path, sep='', dtype=dtype)

#=========================================================================================
def initialize(ck='reconstructed', planets=None, asof=None,
               spk='reconstructed', gapfill=True,
               mst_pck=True, irregulars=True):
    """Initialize key information about the VIMS instrument.

    Must be called first. After the first call, later calls to this function are ignored.

    Input:
        ck,spk      'predicted', 'reconstructed', or 'none', depending on
                    which kernels are to be used. Defaults are 'reconstructed'.
                    Use 'none' if the kernels are to be managed manually.
        planets     A list of planets to pass to define_solar_system. None or
                    0 means all.
        asof        Only use SPICE kernels that existed before this date;
                    None to ignore.
        gapfill     True to include gapfill CKs. False otherwise.
        mst_pck     True to include MST PCKs, which update the rotation models
                    for some of the small moons.
        irregulars  True to include the irregular satellites;
                    False otherwise.
    """
    UVIS.initialize(ck=ck, planets=planets, asof=asof, spk=spk, gapfill=gapfill,
                    mst_pck=mst_pck, irregulars=irregulars)

#=========================================================================================
class UVIS(object):
    """An instance-free class to hold Cassini UVIS instrument parameters."""

    instrument_kernel = None
    fovs = {}
    initialized = False

    # Map NAIF body names (following "CASSINI_UVIS_") to (detector, resolution)
    abbrevs = {'CASSINI_UVIS_FUV_HI' : ('FUV', 'HIGH_RESOLUTION'),
               'CASSINI_UVIS_FUV_LO' : ('FUV', 'LOW_RESOLUTION'),
               'CASSINI_UVIS_FUV_OCC': ('FUV', 'OCCULTATION'),
               'CASSINI_UVIS_EUV_HI' : ('EUV', 'HIGH_RESOLUTION'),
               'CASSINI_UVIS_EUV_LO' : ('EUV', 'LOW_RESOLUTION'),
               'CASSINI_UVIS_EUV_OCC': ('EUV', 'OCCULTATION'),
               'CASSINI_UVIS_SOLAR'  : ('SOLAR',   ''),
               'CASSINI_UVIS_SOL_OFF': ('SOL_OFF', ''),
               'CASSINI_UVIS_HSP'    : ('HSP',     ''),
               'CASSINI_UVIS_HDAC'   : ('HDAC',    '')}

    # Map detector to NAIF frame ID
    frame_ids = {'FUV'    : 'CASSINI_UVIS_FUV',
                 'EUV'    : 'CASSINI_UVIS_EUV',
                 'SOLAR'  : 'CASSINI_UVIS_SOLAR',
                 'SOL_OFF': 'CASSINI_UVIS_SOL_OFF',
                 'HSP'    : 'CASSINI_UVIS_HSP',
                 'HDAC'   : 'CASSINI_UVIS_HDAC'}

    #=====================================================================================
    @staticmethod
    def initialize(ck='reconstructed', planets=None, asof=None, spk='reconstructed',
                   gapfill=True, mst_pck=True, irregulars=True):
        """Fill in key information about the UVIS channels.

        Must be called first. After the first call, later calls to this function are
        ignored.

        Input:
            ck,spk      'predicted', 'reconstructed', or 'none', depending on which
                        kernels are to be used. Defaults are 'reconstructed'. Use 'none'
                        if the kernels are to be managed manually.
            planets     A list of planets to pass to define_solar_system. None or 0 means
                        all.
            asof        Only use SPICE kernels that existed before this date; None to
                        ignore.
            gapfill     True to include gapfill CKs. False otherwise.
            mst_pck     True to include MST PCKs, which update the rotation models for
                        some of the small moons.
            irregulars  True to include the irregular satellites; False otherwise.
        """

        # Quick exit after first call
        if UVIS.initialized:
            return

        Cassini.initialize(ck=ck, planets=planets, asof=asof, spk=spk, gapfill=gapfill,
                           mst_pck=mst_pck, irregulars=irregulars)
        Cassini.load_instruments(asof=asof)

        # Load the instrument kernel
        UVIS.instrument_kernel = Cassini.spice_instrument_kernel('UVIS')[0]

        # TEMPORARY FIX to a loader problem
        ins = UVIS.instrument_kernel['INS']
        ins['CASSINI_UVIS_FUV_HI']  = ins[-82840]
        ins['CASSINI_UVIS_FUV_LO']  = ins[-82841]
        ins['CASSINI_UVIS_FUV_OCC'] = ins[-82842]
        ins['CASSINI_UVIS_EUV_HI']  = ins[-82843]
        ins['CASSINI_UVIS_EUV_LO']  = ins[-82844]
        ins['CASSINI_UVIS_EUV_OCC'] = ins[-82845]
        ins['CASSINI_UVIS_HSP']     = ins[-82846]
        ins['CASSINI_UVIS_HDAC']    = ins[-82847]
        ins['CASSINI_UVIS_SOLAR']   = ins[-82848]
        ins['CASSINI_UVIS_SOL_OFF'] = ins[-82848]

        # Construct a flat FOV and load the frame for each detector
        for key in UVIS.abbrevs.keys():
            (detector, resolution) = UVIS.abbrevs[key]

            # Construct the SpiceFrame
            ignore = oops.frame.SpiceFrame(UVIS.frame_ids[detector])

            # Get the FOV angles
            info = UVIS.instrument_kernel['INS'][key]

            if info['FOV_SHAPE'] == 'RECTANGLE':
                u_angle = 2. * info['FOV_CROSS_ANGLE'] * oops.RPD
                v_angle = 2. * info['FOV_REF_ANGLE'] * oops.RPD
            elif info['FOV_SHAPE'] == 'CIRCLE':
                u_angle = 2. * info['FOV_REF_ANGLE'] * oops.RPD
                v_angle = u_angle
            else:
                raise ValueError('Unrecognized FOV_SHAPE: ' + info['FOV_SHAPE'])

            # Define the frame for 1 or 64 lines
            # Not every combination is really used but that doesn't matter
            for lines in {1, 64}:
                fov = oops.fov.FlatFOV((u_angle, v_angle/lines), (1,lines))
                UVIS.fovs[(detector, resolution, lines)] = fov

        UVIS.initialized = True

    #=====================================================================================
    @staticmethod
    def reset():
        """Resets the internal Cassini UVIS parameters. Can be useful for
        debugging.
        """

        UVIS.instrument_kernel = None
        UVIS.fovs = {}
        UVIS.initialized = False

        Cassini.reset()

##########################################################################################
