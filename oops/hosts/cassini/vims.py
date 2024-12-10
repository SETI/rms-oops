##########################################################################################
# oops/hosts/cassini/vims.py
#
# Known shortcomings:
#
# For SAMPLING_MODE_ID == "UNDER" (aka Nyquist sampling), the FOV boundary will be
# be slightly off because an FOV object cannot handle overlapping, double-sized pixels.
# The proper boundary is 1/64th larger along the line direction to account for the larger
# pixel.
##########################################################################################

import numpy as np
from numpy.lib import stride_tricks

import julian
import pdsparser
import cspyce
import oops

from oops.hosts.cassini import Cassini
from oops.hosts.pds3    import pds3

from filecache import FCPath

# Timing correction as of 1/9/15 -- MRS
# EXTRA_INTERSAMPLE_DELAY =  0.000363     # observed empirically in V1555349441
# TIME_CORRECTION         = -0.337505

TIME_FACTOR = 1.01725

# From Matt Hedman's IDL program navims.pro
#
# vims_params=fltarr(6)
# vims_params(0)=0.495e-3 ;pixel size of IR channels (rad) ;070910
# vims_params(1)=0.506e-3 ; pixel size of VIS channels (rad)
# vims_params(2)=-1.7 ; VIS offset x (in pixels)
# vims_params(3)=+1.5 ; VIS offset z (in pixels)
# vims_params(4)=1.98; HI-RES IR
# vims_params(5)=2.99; HI-RES VIS
#
# xi=(xo-1+i-31.5)*vims_params[0]
# zi=(zo-1+j-31.5)*vims_params[0]
# xv=(xo-1+i-31.5+vims_params[2])*vims_params[1]
# zv=(zo-1+j-31.5+vims_params[3])*vims_params[1]
#
# if hires(0) eq 1 then begin
#     hrf=vims_params(4)
#     xi=(xo-1+sq(1)/2./hrf+i/hrf-31.5)*vims_params[0]
# end
#
# if hires(1) eq 1 then begin
#     hrf=vims_params(5)
#     xv=(xo-1+sq(1)/hrf+i/hrf-31.5 +vims_params[2])*vims_params[1]
#     zv=(zo-1-0+sq(3)/hrf+j/hrf-31.5 +vims_params[3])*vims_params[1]
# end
#
# aimpointi=[xi,zi]
# aimpointv=[xv,zv]

IR_NORMAL_PIXEL  = 0.495e-3
VIS_NORMAL_PIXEL = 0.506e-3

IR_HIRES_FACTOR  = 1.98
VIS_HIRES_FACTOR = 2.99

IR_NORMAL_SCALE  = oops.Pair((IR_NORMAL_PIXEL,  IR_NORMAL_PIXEL))
VIS_NORMAL_SCALE = oops.Pair((VIS_NORMAL_PIXEL, VIS_NORMAL_PIXEL))

IR_HIRES_SCALE  = oops.Pair((IR_NORMAL_PIXEL/IR_HIRES_FACTOR, IR_NORMAL_PIXEL))
VIS_HIRES_SCALE = oops.Pair((VIS_NORMAL_PIXEL/VIS_HIRES_FACTOR,
                             VIS_NORMAL_PIXEL/VIS_HIRES_FACTOR))

IR_OVER_VIS = IR_NORMAL_PIXEL / VIS_NORMAL_PIXEL

IR_FULL_FOV  = oops.fov.FlatFOV(IR_NORMAL_SCALE,  oops.Pair((64,64)))
VIS_FULL_FOV = oops.fov.FlatFOV(VIS_NORMAL_SCALE, oops.Pair((64,64)))

##########################################################################################
# FMT files have fixed values
##########################################################################################

CORE_DESCRIPTION_FMT = pds3.fast_dict("""\
  CORE_ITEM_BYTES                = 2
  CORE_ITEM_TYPE                 = SUN_INTEGER
  CORE_BASE                      = 0.0
  CORE_MULTIPLIER                = 1.0
  CORE_VALID_MINIMUM             = -4095
  CORE_NULL                      = -8192
  CORE_LOW_REPR_SATURATION       = -32767
  CORE_LOW_INSTR_SATURATION      = -32766
  CORE_HIGH_REPR_SATURATION      = -32764
  CORE_HIGH_INSTR_SATURATION     = -32765
  CORE_MINIMUM_DN                = -122
  CORE_NAME                      = "RAW DATA NUMBER"
  CORE_UNIT                      = DIMENSIONLESS
""")

SUFFIX_DESCRIPTION_FMT = pds3.fast_dict("""\
  GROUP                          = SAMPLE_SUFFIX
    SUFFIX_NAME                  = BACKGROUND
    SUFFIX_UNIT                  = DIMENSIONLESS
    SUFFIX_ITEM_BYTES            = 4
    SUFFIX_ITEM_TYPE             = SUN_INTEGER
    SUFFIX_BASE                  = 0.0
    SUFFIX_MULTIPLIER            = 1.0
    SUFFIX_VALID_MINIMUM         = 0
    SUFFIX_NULL                  = -8192
    SUFFIX_LOW_REPR_SAT          = -32767
    SUFFIX_LOW_INSTR_SAT         = -32766
    SUFFIX_HIGH_REPR_SAT         = -32764
    SUFFIX_HIGH_INSTR_SAT        = -32765
  END_GROUP                      = SAMPLE_SUFFIX

  GROUP                          = BAND_SUFFIX
    SUFFIX_NAME                  = (X_SCAN_DRIVE_CURRENT,
                                    Z_SCAN_DRIVE_CURRENT,
                                    X_SCAN_MIRROR_POSITION,
                                    Z_SCAN_MIRROR_POSITION)
    SUFFIX_UNIT                  = (DIMENSIONLESS,DIMENSIONLESS,
                                    DIMENSIONLESS,DIMENSIONLESS)
    SUFFIX_ITEM_TYPE             = (SUN_INTEGER,SUN_INTEGER,
                                    SUN_INTEGER,SUN_INTEGER)
    SUFFIX_ITEM_BYTES            = (4,4,4,4)
    SUFFIX_BASE                  = (0.0,0.0,0.0,0.0)
    SUFFIX_MULTIPLIER            = (1.0,1.0,1.0,1.0)
    SUFFIX_VALID_MINIMUM         = (0,0,0,0)
    SUFFIX_NULL                  = (-8192,-8192,-8192,-8192)
    SUFFIX_LOW_REPR_SAT          = (-32767,-32767,-32767,-32767)
    SUFFIX_LOW_INSTR_SAT         = (-32766,-32766,-32766,-32766)
    SUFFIX_HIGH_INSTR_SAT        = (-32765,-32765,-32765,-32765)
    SUFFIX_HIGH_REPR_SAT         = (-32764,-32764,-32764,-32764)
  END_GROUP                      = BAND_SUFFIX
""")

BAND_BIN_CENTER_FMT = pds3.fast_dict("""\
  GROUP                          = BAND_BIN
    BAND_BIN_CENTER = (0.35,0.36,0.37,0.37,0.38,0.39,0.40,0.40,0.41,0.42,
      0.42,0.43,0.44,0.45,0.45,0.46,0.47,0.48,0.49,0.49,0.50,0.51,0.51,0.52,
      0.53,0.53,0.54,0.55,0.56,0.56,0.57,0.58,0.59,0.59,0.60,0.61,0.62,0.62,
      0.63,0.64,0.64,0.65,0.66,0.67,0.67,0.68,0.69,0.70,0.70,0.71,0.72,0.72,
      0.73,0.74,0.75,0.75,0.76,0.77,0.78,0.78,0.79,0.80,0.81,0.81,0.82,0.83,
      0.83,0.84,0.85,0.86,0.86,0.87,0.88,0.89,0.89,0.90,0.91,0.92,0.92,0.93,
      0.94,0.94,0.95,0.96,0.97,0.97,0.98,0.99,1.00,1.00,1.01,1.02,1.02,1.03,
      1.04,1.05,0.863,0.879,0.896,0.912,0.928,0.945,0.961,0.977,0.994,1.010,
      1.026,1.043,1.060,1.077,1.093,1.109,1.125,1.142,1.159,1.175,1.191,1.207,
      1.224,1.240,1.257,1.273,1.290,1.306,1.322,1.338,1.355,1.372,1.388,1.404,
      1.421,1.437,1.453,1.470,1.487,1.503,1.519,1.535,1.552,1.569,1.585,1.597,
      1.620,1.637,1.651,1.667,1.684,1.700,1.717,1.733,1.749,1.766,1.783,1.799,
      1.815,1.831,1.848,1.864,1.882,1.898,1.914,1.930,1.947,1.964,1.980,1.997,
      2.013,2.029,2.046,2.063,2.079,2.095,2.112,2.128,2.145,2.162,2.178,2.194,
      2.211,2.228,2.245,2.261,2.277,2.294,2.311,2.328,2.345,2.363,2.380,2.397,
      2.413,2.430,2.446,2.462,2.479,2.495,2.512,2.528,2.544,2.559,2.577,2.593,
      2.610,2.625,2.642,2.656,2.676,2.691,2.707,2.728,2.743,2.758,2.776,2.794,
      2.811,2.827,2.845,2.861,2.877,2.894,2.910,2.926,2.942,2.958,2.972,2.996,
      3.009,3.025,3.043,3.059,3.075,3.092,3.107,3.125,3.142,3.158,3.175,3.192,
      3.209,3.227,3.243,3.261,3.278,3.294,3.311,3.328,3.345,3.361,3.377,3.394,
      3.410,3.427,3.444,3.460,3.476,3.493,3.508,3.525,3.542,3.558,3.575,3.591,
      3.609,3.626,3.644,3.660,3.678,3.695,3.712,3.729,3.746,3.763,3.779,3.796,
      3.812,3.830,3.846,3.857,3.877,3.894,3.910,3.926,3.943,3.959,3.975,3.992,
      4.008,4.024,4.042,4.058,4.076,4.092,4.110,4.127,4.144,4.161,4.178,4.193,
      4.206,4.219,4.237,4.255,4.273,4.292,4.310,4.328,4.346,4.361,4.378,4.393,
      4.410,4.427,4.443,4.461,4.477,4.495,4.511,4.529,4.547,4.563,4.581,4.598,
      4.615,4.631,4.649,4.665,4.682,4.698,4.715,4.732,4.749,4.765,4.782,4.798,
      4.815,4.831,4.848,4.864,4.881,4.898,4.915,4.932,4.949,4.967,4.984,5.001,
      5.017,5.036,5.052,5.069,5.086,5.102)
    BAND_BIN_UNIT                = MICROMETER
    BAND_BIN_ORIGINAL_BAND = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,
      19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,
      43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,
      67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,
      91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,
      111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,
      129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,
      147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,
      165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,
      183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,
      201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,
      219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,
      237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,
      255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,
      273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,
      291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,
      309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,
      327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,
      345,346,347,348,349,350,351)
  END_GROUP                      = BAND_BIN
""")

##########################################################################################
# Standard class methods
##########################################################################################

def from_file(filespec, data=True):
    """A general, static method to return a pair of Observation objects based on a given
    Cassini VIMS data file or label file.

    Input:
        filespec        the full path to a VIMS cube file or its PDS label.
        data            if True, data arrays are included in the returned observation
                        objects. Use a tuple of two booleans to specify whether to include
                        the VIS and IR data independently.

    Return:             (vis, ir)
        vis             the VIS observation, or None if the VIS channel was inactive.
        ir              the IR observation, or None if the IR channel was inactive.
    """

    VIMS.initialize()   # Define everything the first time through; use defaults
                        # unless initialize() is called explicitly.

    if not isinstance(data, (tuple, list)):
        data = (data, data)

    filespec = FCPath(filespec)

    label = pds3.fast_dict(filespec)

    # Insert "data_file" and "header_recs"
    # Convert ISIS .qub info to a standard PDS3 label dictionary
    if 'QUBE' in label:

        # PDS3 labels use "SPECTRAL_QUBE" in place of "QUBE"
        label['SPECTRAL_QUBE'] = label['QUBE']

        # Copy the non-structural issue to the top-level dictionary
        for key, value in label['QUBE'].items():
            if (key in {'AXES', 'AXIS_NAME'}
                or key[:5] == 'CORE_'
                or key[:7] in {'SUFFIX_', 'SAMPLE_', 'BAND_SU'}):
                    continue
            label[key] = value

        # Rename "PACKING" to "PACKING_FLAG"
        label['PACKING_FLAG'] = label['PACKING']

        # Add these items
        label['data_file'] = filespec
        label['header_recs'] = label['^QUBE'] - 1

    # Otherwise, convert PDS3 label info to a standard ISIS dictionary
    else:

        # Insert the needed info from the .FMT files
        label['SPECTRAL_QUBE'].update(CORE_DESCRIPTION_FMT)
        label['SPECTRAL_QUBE'].update(SUFFIX_DESCRIPTION_FMT)
        label.update(BAND_BIN_CENTER_FMT)

        # ISIS labels use "QUBE" in place of "SPECTRAL_QUBE"
        label['QUBE'] = label['SPECTRAL_QUBE']

        # Add these items
        label['data_file'] = filespec.with_name(label['^QUBE'][0])
        label['header_recs'] = label['^QUBE'][1] - 1

    qube_dict = label['SPECTRAL_QUBE']

    # Load any needed SPICE kernels
    tstart = julian.tdb_from_tai(julian.tai_from_iso(label['START_TIME']))
    Cassini.load_cks( tstart, tstart + 3600.)
    Cassini.load_spks(tstart, tstart + 3600.)

    # Check power state of each channel: [0] is IR; [1] is VIS
    ir_is_off  = label['POWER_STATE_FLAG'][0] == 'OFF'
    vis_is_off = label['POWER_STATE_FLAG'][1] == 'OFF'

    ########################################
    # Load the data arrays
    ########################################

    (samples, bands, lines) = qube_dict['CORE_ITEMS']
    assert bands in (352, 256, 96)

    vis_data = None
    ir_data = None
    times = None

    if data[0] or data[1]:
        qub_file = filespec.with_suffix('.qub')
        (array, times) = _load_data_and_times(qub_file, label)
        assert array.shape == (lines, samples, bands), 'incorrect array shape'

        if bands == 352:
            vis_data = array[:,:,:96]   # index order is [line, sample, band]
            ir_data  = array[:,:,96:]
        elif bands == 256:              # only happens in a few early cubes
            vis_data = None
            ir_data = array
        else:
            vis_data = array
            ir_data = None

    ########################################
    # Define the FOVs
    ########################################

    swath_width = label['SWATH_WIDTH']
    swath_length = label['SWATH_LENGTH']

    frame_size = swath_width * swath_length
    frames = (samples * lines) // frame_size
    assert samples * lines == frames * frame_size

    # Replace multiple one-line frames by one frame, multiple lines
    if frames > 1 and frame_size != 1:
        assert swath_length == 1
        swath_length = frames
        lines = frames
        frame_size = swath_width * swath_length

    uv_shape = (swath_width, swath_length)

    x_offset = label['X_OFFSET']
    z_offset = label['Z_OFFSET']
    uv_los = (33. - x_offset, 33. - z_offset)

    vis_sampling = label['SAMPLING_MODE_ID'][1].strip()  # handle ' NORMAL'
    ir_sampling  = label['SAMPLING_MODE_ID'][0].strip()

    # VIS FOV
    if vis_sampling == 'HI-RES':
        vis_fov = oops.fov.FlatFOV(VIS_HIRES_SCALE, uv_shape,
                                   (VIS_HIRES_FACTOR * uv_los[0] - uv_shape[0],
                                    VIS_HIRES_FACTOR * uv_los[1] - uv_shape[1]))

    elif uv_shape == (64,64):
        vis_fov = VIS_FULL_FOV

    else:
        vis_fov = oops.fov.FlatFOV(VIS_NORMAL_SCALE, uv_shape,
                                   (IR_OVER_VIS * uv_los[0], IR_OVER_VIS * uv_los[1]))

    # IR FOV
    if label['INSTRUMENT_MODE_ID'] == 'OCCULTATION':
        if ir_sampling == 'NORMAL':
            ir_fov = oops.fov.FlatFOV(IR_NORMAL_SCALE, uv_shape, uv_los)
        else:
            ir_fov = oops.fov.FlatFOV(IR_HIRES_SCALE, uv_shape, uv_los)

    elif ir_sampling in ('HI-RES','UNDER'):
        ir_fov = oops.fov.FlatFOV(IR_HIRES_SCALE, uv_shape,
                                  (IR_HIRES_FACTOR * uv_los[0] - uv_shape[0]/2.,
                                   uv_los[1]))

    elif uv_shape == (64,64):
        ir_fov = IR_FULL_FOV

    else:
        ir_fov = oops.fov.FlatFOV(IR_NORMAL_SCALE, uv_shape, uv_los)

    # Nyquist sampling
    if ir_sampling == 'UNDER':
        ### TBD: VIMS IR sampling mode UNDER is untested!!
        # Use ir_det_size = 2 for RasterSlit1D and RasterSlit observations
        # Use (1., ir_det_size) for RasterScan observations
        ir_fov = oops.fov.GapFOV(IR_NORMAL_SCALE, uv_shape, uv_los)

    else:
        ir_fov = oops.fov.FlatFOV(IR_NORMAL_SCALE, uv_shape, uv_los)

    ########################################
    # Define the cadences
    ########################################

    # Define cadences based on header parameters
    ir_texp  = label['EXPOSURE_DURATION'][0] * 0.001 * TIME_FACTOR
    vis_texp = label['EXPOSURE_DURATION'][1] * 0.001 * TIME_FACTOR
    vis_texp_nonzero = max(vis_texp, 1.e-8) # avoids divide-by-zero in cadences

    interframe_delay = label['INTERFRAME_DELAY_DURATION'] * 0.001 * TIME_FACTOR
    interline_delay  = label['INTERLINE_DELAY_DURATION']  * 0.001 * TIME_FACTOR

    # Adjust the timing of one line, multiple frames
    if frames > 1 and frame_size != 1:
        interline_delay = interframe_delay

    length_stride = max(ir_texp * swath_width, vis_texp) + interline_delay

    backplane_cadence = None

    # Define a cadence based on the time backplane, if it is present
    if times is None:
        pass

    elif label['OVERWRITTEN_CHANNEL_FLAG'] == 'ON':
        times = times.ravel()
        assert times[0] < times[1]
        assert vis_is_off
        backplane_cadence = oops.cadence.Sequence(times, ir_texp)

    elif label['PACKING_FLAG'] == 'ON':
        times = times.ravel()
        assert times[0] == times[1]
        tstart = times[0]
        frame_cadence = oops.cadence.Sequence(times[::frame_size], texp=0.)

    else:       # No packing plus no embedded timing just means a better tstart
        times = times.ravel()
        assert times[0] == times[1]
        tstart = times[0]

    vis_header_cadence = oops.cadence.Metronome(tstart, length_stride, vis_texp_nonzero,
                                                swath_length)
    ir_fast_cadence = oops.cadence.Metronome(tstart, ir_texp, ir_texp, swath_width)
    ir_header_cadence = oops.cadence.DualCadence(vis_header_cadence, ir_fast_cadence)

    # At this point...
    #   vis_header_cadence  always defined, always 1-D.
    #   ir_fast_cadence     always defined, always 1-D.
    #   ir_header_cadence   always defined, always 2-D.
    #   backplane_cadence   defined if timing was recorded, always 1-D.

    ########################################
    # Define the coordinate frames
    ########################################

    vis_frame_id = 'CASSINI_VIMS_V'
    ir_frame_id  = 'CASSINI_VIMS_IR'

    if (label['TARGET_NAME'] == 'SUN' or '_SOL' in label['OBSERVATION_ID']):
        ir_frame_id = 'CASSINI_VIMS_IR_SOL'

    ########################################
    # Construct the Observation objects
    ########################################

    vis_obs = None
    ir_obs  = None

    # POINT/OCCULTATION case
    if swath_width == 1 and swath_length == 1:
        assert vis_is_off
        if backplane_cadence is None:
            fast_stride = ir_texp   # + EXTRA_INTERSAMPLE_DELAY
            fastcad = oops.cadence.Metronome(tstart, fast_stride, ir_texp, samples)

            slow_stride = ir_texp * samples + interline_delay
            slowcad = oops.cadence.Metronome(tstart, slow_stride, slow_stride, lines)
            fullcad = oops.cadence.DualCadence(slowcad, fastcad)
            ir_cadence = oops.cadence.ReshapedCadence(fullcad, (samples*lines,))

        else:
            ir_cadence = backplane_cadence

        if ir_data is not None:
            ir_data = ir_data.reshape((frames, 256))

        ir_obs = oops.obs.Pixel(('t','b'), ir_cadence, ir_fov, 'CASSINI', ir_frame_id)

    # Single LINE case
    elif swath_length == 1:
        if not vis_is_off:
            if vis_data is not None:
                vis_data = vis_data.reshape((samples, 96))

            vis_obs = oops.obs.Slit1D(('u','b'), tstart, vis_texp_nonzero, vis_fov,
                                       'CASSINI', vis_frame_id)

        if not ir_is_off:
            if ir_data is not None:
                ir_data = ir_data.reshape((samples, 256))

            if backplane_cadence is not None:
                ir_fast_cadence = backplane_cadence

            ir_obs = oops.obs.RasterSlit1D(('ut','b'), ir_fast_cadence, ir_fov,
                                           'CASSINI', ir_frame_id)

    # Single 2-D IMAGE case
    elif samples == swath_width and lines == swath_length:
        if not vis_is_off:
            vis_obs = oops.obs.TimedImage(('vt','u','b'), vis_header_cadence, vis_fov,
                                          'CASSINI', vis_frame_id)

        if not ir_is_off:
            if backplane_cadence is None:
                ir_cadence = oops.cadence.DualCadence(vis_header_cadence,
                                                      ir_fast_cadence)
            else:
                ir_cadence = oops.cadence.ReshapedCadence(backplane_cadence,
                                                         (lines,samples))

            ir_obs = oops.obs.TimedImage(('vslow','ufast','b'), ir_cadence, ir_fov,
                                         'CASSINI', ir_frame_id)

    # Multiple LINE case
    elif swath_length == 1 and swath_length == lines:
        if not vis_is_off:
            vis_obs = oops.obs.TimedImage(('vt','u','b'), frame_cadence, vis_fov,
                                          'CASSINI', vis_frame_id)

        if not ir_is_off:
            if backplane_cadence is None:
                ir_cadence = oops.cadence.DualCadence(frame_cadence, ir_fast_cadence)
            else:
                ir_cadence = oops.cadence.ReshapedCadence(backplane_cadence,
                                                         (lines,samples))

            ir_obs = oops.obs.TimedImage(('vslow','ufast','b'), ir_cadence, ir_fov,
                                         'CASSINI', ir_frame_id)

# 1/9/15 broken code no longer needed
#     # Multiple 2-D IMAGE case
#
#     elif lines == frames and samples == swath_width * swath_length:
#         if not vis_is_off:
#
#             # Reshape the data array
#             if vis_data is not None:
#                 vis_data = vis_data.reshape(frames, swath_length, swath_width,
#                                 vis_data.shape[-1])
#
#             # Define the first 2-D pushbroom observation
#             vis_first_obs = oops.obs.TimedImage(('t', 'vt','u','b'), (1.,1.),
#                                 vis_header_cadence, vis_fov,
#                                 'CASSINI', vis_frame_id)
#
#             # Define the movie
#             movie_cadence = oops.cadence.DualCadence(frame_cadence,
#                                 vis_header_cadence)
#
#             vis_obs = oops.obs.Movie(('t','vt','u','b'), vis_first_obs,
#                                 movie_cadence)
#
#         if not ir_is_off:
#
#             # Reshape the data array
#             if ir_data is not None:
#                 ir_data = ir_data.reshape(frames, swath_length, swath_width,
#                                           ir_data.shape[-1])
#
#             # Define the first 2-D raster-scan observation
#             ir_first_obs = oops.obs.TimedImage(('vslow','ufast','b'),
#                                 (1., ir_det_size),
#                                 ir_first_cadence, ir_fov,
#                                 'CASSINI', ir_frame_id)
#
#             # Define the 3-D cadence
#             if backplane_cadence is None:
#                 ir_first_cadence = oops.cadence.DualCadence(vis_header_cadence,
#                                 ir_fast_cadence)
#
#                 ir_cadence = oops.cadence.DualCadence(frame_cadence,
#                                 ir_first_cadence)
#             else:
#                 ir_cadence = oops.cadence.ReshapedCadence(backplane_cadence,
#                                 (frames,lines,samples))
#
#             # Define the movie
#             ir_obs = oops.obs.Movie(('t','vslow','ufast','b'), ir_first_obs,
#                                 ir_cadence)
#
#             # Reshape the data array
#             if ir_data is not None:
#                 ir_data = ir_data.reshape(frames, swath_length, swath_width,
#                                           ir_data.shape[-1])
#
#             # Define the first 2-D raster-scan observation
#             ir_first_obs = oops.obs.TimedImage(('vslow','ufast','b'),
#                                 (1., ir_det_size),
#                                 ir_first_cadence, ir_fov,
#                                 'CASSINI', ir_frame_id)
#
#             # Define the 3-D cadence
#             if backplane_cadence is None:
#                 ir_first_cadence = oops.cadence.DualCadence(vis_header_cadence,
#                                 ir_fast_cadence)
#
#                 ir_cadence = oops.cadence.DualCadence(frame_cadence,
#                                 ir_first_cadence)
#             else:
#                 ir_cadence = oops.cadence.ReshapedCadence(backplane_cadence,
#                                 (frames,lines,samples))
#
#             # Define the movie
#             ir_obs = oops.obs.Movie(('t','vslow','ufast','b'), ir_first_obs,
#                                 ir_cadence)

    else:
        raise ValueError(f'unsupported VIMS format in file {filespec}')

    # Insert the data array
    if vis_obs is not None:
        vis_obs.insert_subfield('instrument', 'VIMS')
        vis_obs.insert_subfield('detector', 'VIS')
        vis_obs.insert_subfield('sampling', vis_sampling)
        vis_obs.insert_subfield('dict', label)
        vis_obs.insert_subfield('index_dict', label)# for backward compatibility

        if vis_data is not None and data[0]:
            vis_obs.insert_subfield('data', vis_data)

    if ir_obs is not None:
        ir_obs.insert_subfield('instrument', 'VIMS')
        ir_obs.insert_subfield('detector', 'IR')
        ir_obs.insert_subfield('sampling', ir_sampling)
        ir_obs.insert_subfield('dict', label)
        ir_obs.insert_subfield('index_dict', label)# for backward compatibility

        if ir_data is not None and data[1]:
            ir_obs.insert_subfield('data', ir_data)

    return (vis_obs, ir_obs)

#=========================================================================================
def _load_data_and_times(filespec, label):
    """Load the data array from the file.

    If time backplanes are present, also return an array of times in seconds TDB as
    derived from these backplanes.

    Input:
        filespec        full path to the data file.
        label           the label dictionary.

    Return:             (data, times)
        data            a Numpy array containing the data in axis order (line, sample,
                        band).
        times           the time sampling array in (line, sample) axis order, or None if
                        no time backplane is found in the file.

    Note: This procedure is absurdly complicated but it has been rather carefully
    debugged. --MRS 7/4/12.
    """

    qube_dict = label['SPECTRAL_QUBE']

    # Extract key parameters from the file header
    core_items   = qube_dict['CORE_ITEMS']
    core_samples = core_items[0]
    core_bands   = core_items[1]
    core_lines   = core_items[2]
    core_item_bytes = qube_dict.get('CORE_ITEM_BYTES', 2)
    core_item_type  = qube_dict.get('CORE_ITEM_TYPE', 'SUN_INTEGER')

    sample_suffix_items = qube_dict['SUFFIX_ITEMS'][0]
    band_suffix_items   = qube_dict['SUFFIX_ITEMS'][1]

    suffix_item_bytes = 4

    key = 'SAMPLE_SUFFIX_ITEM_TYPE' if sample_suffix_items else 'BAND_SUFFIX_ITEM_TYPE'
    suffix_item_type = qube_dict.get(key, 'SUN_INTEGER')
    if isinstance(suffix_item_type, (list, tuple)):
        suffix_item_type = suffix_item_type[0]  # all backplanes/sideplanes are same type

    record_bytes = label['RECORD_BYTES']
    header_bytes = record_bytes * label['header_recs']

    # Make sure we have byte-aligned values
    assert (core_samples * core_item_bytes) % suffix_item_bytes == 0, 'misaligned items'

    ############################

    # Determine the dtype and strides for the core item array
    band_stride = (core_samples * core_item_bytes +
                   sample_suffix_items * suffix_item_bytes)

    core_items_in_line = core_samples * core_bands
    suffix_items_in_line = ((core_samples + sample_suffix_items) *
                            (core_bands   + band_suffix_items)
                            - core_items_in_line)
    line_stride = (core_items_in_line * core_item_bytes +
                   suffix_items_in_line * suffix_item_bytes)

    # Locate the cube data in the file in units of core_item_bytes
    offset = header_bytes // core_item_bytes
    size = line_stride * core_lines

    # Determine the dtype for the file core
    if 'SUN_' in core_item_type or 'MSB_' in core_item_type:
        core_dtype = '>'
    elif 'PC_' in core_item_type or  'LSB_' in core_item_type:
        core_dtype = '<'
    else:
        raise TypeError('Unrecognized byte order: ' + core_item_type)

    if 'UNSIGNED' in core_item_type:
        core_dtype += 'u'
        native_dtype = 'int'
    elif 'INTEGER' in core_item_type:
        core_dtype += 'i'
        native_dtype = 'int'
    elif 'REAL'    in core_item_type:
        core_dtype += 'f'
        native_dtype = 'float'
    else:
        raise TypeError('Unrecognized core data type: ' + core_item_type)

    core_dtype += str(core_item_bytes)

    # Read the file as core dtypes
    local_path = filespec.retrieve()
    array = np.fromfile(local_path, dtype=core_dtype)

    # Slice away the core lines, leaving off the line suffix
    array = array[offset:offset+size]

    # Create a data array using new strides in (line, sample, band) order
    data = stride_tricks.as_strided(array,
                                    strides = (line_stride, core_item_bytes, band_stride),
                                    shape   = (core_lines,  core_samples,    core_bands))

    # Convert core to a native 3-D array
    data = data.astype(native_dtype)

    # If there are no time backplanes, we're done
    band_suffix_name = qube_dict['BAND_SUFFIX_NAME']
    if 'SLICE_TIME_SECONDS' not in band_suffix_name:
        return (data, None)

    ############################

    # Determine the dtype for the file core
    if 'SUN_' in core_item_type or 'MSB_' in suffix_item_type:
        suffix_item_dtype = '>'
    elif 'PC_' in core_item_type or  'LSB_' in suffix_item_type:
        suffix_item_dtype = '<'
    else:
        raise TypeError('Unrecognized byte order: ' + suffix_item_type)

    if 'UNSIGNED' in suffix_item_type:
        suffix_item_dtype += 'u'
        native_dtype = 'int'
    elif 'INTEGER' in suffix_item_type:
        suffix_item_dtype += 'i'
        native_dtype = 'int'
    elif 'REAL'    in suffix_item_type:
        suffix_item_dtype += 'f'
        native_dtype = 'float'
    else:
        raise TypeError('Unrecognized suffix data type: ' + suffix_item_type)

    suffix_item_dtype += str(suffix_item_bytes)

    # The offset array skips over the first (bands,samples) to begin at the
    # memory location of the first band suffix backplane
    offset_array = np.frombuffer(array.data,
                offset = core_bands * (core_samples * core_item_bytes +
                                       sample_suffix_items * suffix_item_bytes),
                dtype = suffix_item_dtype)

    # Extract the band suffix array using new strides in
    # (backplane, line, sample) order
    backplane_stride = suffix_item_bytes * (core_samples + sample_suffix_items)
    backplane = stride_tricks.as_strided(offset_array,
                            strides = (backplane_stride,  line_stride, suffix_item_bytes),
                            shape   = (band_suffix_items, core_lines,  core_samples))

    # Convert to spacecraft clock
    seconds = backplane[band_suffix_name.index('SLICE_TIME_SECONDS')]
    ticks = backplane[band_suffix_name.index('SLICE_TIME_TICKS')]
    sclock = seconds + ticks/15959.

    # Convert to TDB
    mask = (seconds == -8192)       # Sometimes all are -8192 except first
    if np.any(mask):
        assert np.all(mask.ravel()[1:])
        formatted = '%16.3f' % sclock[0,0]
        times = np.empty(sclock.shape)
        times[...] = cspyce.scs2e(-82, formatted)

    else:
        sclock_min = sclock.min()
        formatted = '%16.3f' % sclock_min
        tdb_min = cspyce.scs2e(-82, formatted) + (sclock_min - float(formatted))

        sclock_max = sclock.max()
        formatted = '%16.3f' % sclock_max
        tdb_max = cspyce.scs2e(-82, formatted) + (sclock_max - float(formatted))

        times = tdb_min + (sclock - sclock_min) * ((tdb_max - tdb_min) /
                                                   (sclock_max - sclock_min))

    return (data, times)

#=========================================================================================
def meshgrid_and_times(obs, oversample=6, extend=1.5):
    """A meshgrid object and time array that oversamples and extends the dimensions of the
    field of view of a VIMS observation.

    Input:
        obs             the VIMS observation object to for which to generate a meshgrid
                        and a time array.
        oversample      the factor by which to oversample the field of view, in units of
                        the full-resolution VIMS pixel size.
        extend          the number of pixels by which to extend the field of view, in
                        units of the oversampled pixel.

    Return:             (mesgrid, time)
    """

    shrinkage = {('IR',  'NORMAL'): (1,1),
                 ('IR',  'HI-RES'): (2,1),
                 ('IR',  'UNDER' ): (2,1),
                 ('VIS', 'NORMAL'): (1,1),
                 ('VIS', 'HI-RES'): (3,3)}

    assert obs.instrument == 'VIMS'

    (ushrink,vshrink) = shrinkage[(obs.detector, obs.sampling)]

    oversample = float(oversample)
    undersample = (ushrink, vshrink)

    ustep = ushrink / oversample
    vstep = vshrink / oversample

    origin = (-extend * ustep, -extend * vstep)

    limit = (obs.fov.uv_shape.vals[0] + extend * ustep,
             obs.fov.uv_shape.vals[1] + extend * vstep)

    meshgrid = oops.Meshgrid.for_fov(obs.fov, origin, undersample, oversample, limit,
                                     swap=True)

    time = obs.uvt(obs.fov.nearest_uv(meshgrid.uv).swapxy())[1]

    return (meshgrid, time)

#=========================================================================================
def initialize(ck='reconstructed', planets=None, asof=None, spk='reconstructed',
               gapfill=True, mst_pck=True, irregulars=True):
    """Initialize key information about the VIMS instrument.

    Must be called first. After the first call, later calls to this function
    are ignored.

    Input:
        ck,spk      'predicted', 'reconstructed', or 'none', depending on which kernels
                    are to be used. Defaults are 'reconstructed'. Use 'none' if the
                    kernels are to be managed manually.
        planets     A list of planets to pass to define_solar_system. None or 0 means all.
        asof        Only use SPICE kernels that existed before this date;
                    None to ignore.
        gapfill     True to include gapfill CKs. False otherwise.
        mst_pck     True to include MST PCKs, which update the rotation models for some of
                    of the small moons.
        irregulars  True to include the irregular satellites; False otherwise.
    """

    VIMS.initialize(ck=ck, planets=planets, asof=asof, spk=spk, gapfill=gapfill,
                    mst_pck=mst_pck, irregulars=irregulars)

##########################################################################################

class VIMS(object):
    """An instance-free class to hold Cassini VIMS instrument parameters."""

    initialized = False
    instrument_kernel = None

    @staticmethod
    def initialize(ck='reconstructed', planets=None, asof=None, spk='reconstructed',
                   gapfill=True, mst_pck=True, irregulars=True):
        """Fill in key information about the VIS and IR channels.

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
        if VIMS.initialized:
            return

        Cassini.initialize(ck=ck, planets=planets, asof=asof, spk=spk, gapfill=gapfill,
                           mst_pck=mst_pck, irregulars=irregulars)
        Cassini.load_instruments(asof=asof)

        # Load the instrument kernel
        VIMS.instrument_kernel = Cassini.spice_instrument_kernel('VIMS')[0]

        # Construct a SpiceFrame for each detector
        ignore = oops.frame.SpiceFrame('CASSINI_VIMS_V')
        ignore = oops.frame.SpiceFrame('CASSINI_VIMS_IR')
        ignore = oops.frame.SpiceFrame('CASSINI_VIMS_IR_SOL')

        VIMS.initialized = True

    #=====================================================================================
    @staticmethod
    def reset():
        """Reset the internal Cassini VIMS parameters.

        Can be useful for debugging.
        """

        VIMS.instrument_kernel = None
        VIMS.fovs = {}
        VIMS.initialized = False

        Cassini.reset()

##########################################################################################
