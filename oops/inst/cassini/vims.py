################################################################################
# oops/inst/cassini/vims.py
#
# 7/24/12 MRS -- First working version

# Known shortcomings:
#
# For SAMPLING_MODE_ID == "UNDER" (aka Nyquist sampling), the FOV boundary will
# be slightly off because an FOV object cannot handle overlapping, double-sized
# pixels. The proper boundary is 1/64th larger along the line direction to
# account for the larger pixel.
################################################################################

import numpy as np
import numpy.lib.stride_tricks

import tempfile
import pylab
import julian
import pdstable
import pdsparser
import cspice
import oops

from oops.inst.cassini.cassini_ import Cassini

EXTRA_INTERSAMPLE_DELAY =  0.000363     # observed empirically in V1555349441
TIME_CORRECTION         = -0.337505

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

IR_FULL_FOV  = oops.fov.Flat(IR_NORMAL_SCALE,  oops.Pair((64,64)))
VIS_FULL_FOV = oops.fov.Flat(VIS_NORMAL_SCALE, oops.Pair((64,64)))

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, parameters={}):
    """A general, static method to return a list of objects based on a given
    Cassini VIMS image file.

    The method returns a list of tuples, where each tuple consists of the VIS
    observation followed by the IR observation. Most of the time, the list will
    contain only one tuple, but packed arrays and observations done in LINE mode
    are split apart into their constituent objects."""

    VIMS.initialize()   # Define everything the first time through

    # Load the VICAR label
    label = pdsparser.PdsLabel.from_file(filespec).as_dict()

    is_isis_file = "QUBE" in label.keys()
    if is_isis_file:                # If this is an ISIS file...
        info = label["QUBE"]        # ... the info is in the QUBE object
    else:                           # Otherwise, this is a .LBL file
        info = label                # ... and the info is at the top level
        info["CORE_ITEMS"] = label["SPECTRAL_QUBE"]["CORE_ITEMS"]
        info["BAND_SUFFIX_NAME"] = label["SPECTRAL_QUBE"]["BAND_SUFFIX_NAME"]

    # Load any needed SPICE kernels
    tstart = julian.tdb_from_tai(julian.tai_from_iso(info["START_TIME"]))
    Cassini.load_cks( tstart, tstart + 3600.)
    Cassini.load_spks(tstart, tstart + 3600.)

    # Check power state of each channel: [0] is IR; [1] is VIS
    ir_is_off  = info["POWER_STATE_FLAG"][0] == "OFF"
    vis_is_off = info["POWER_STATE_FLAG"][1] == "OFF"

    ########################################
    # Load the data array
    ########################################

    (samples, bands, lines) = info["CORE_ITEMS"]
    assert bands == 352

    if is_isis_file:
        (data, times) = _load_data_and_times(filespec, label)
        assert data.shape == (lines, samples, bands)

        vis_data = data[:,:,:96]    # index order is [line, sample, band]
        ir_data  = data[:,:,96:]

    else:
        vis_data = None
        ir_data = None
        times = None

    ########################################
    # Define the FOVs
    ########################################

    swath_width = info["SWATH_WIDTH"]
    swath_length = info["SWATH_LENGTH"]
    uv_shape = (swath_width, swath_length)

    x_offset = info["X_OFFSET"]
    z_offset = info["Z_OFFSET"]
    uv_los = (33. - x_offset, 33. - z_offset)

    vis_sampling = info["SAMPLING_MODE_ID"][1]
    ir_sampling  = info["SAMPLING_MODE_ID"][0]

    # VIS FOV
    if vis_sampling == "HI-RES":
        vis_fov = oops.fov.Flat(VIS_HIRES_SCALE, uv_shape,
                    (VIS_HIRES_FACTOR * uv_los[0] - uv_shape[0],
                     VIS_HIRES_FACTOR * uv_los[1] - uv_shape[1]))

    elif uv_shape == (64,64):
        vis_fov = VIS_FULL_FOV

    else:
        vis_fov = oops.fov.Flat(VIS_NORMAL_SCALE, uv_shape,
                    (IR_OVER_VIS * uv_los[0], IR_OVER_VIS * uv_los[1]))

    # IR FOV
    if info["INSTRUMENT_MODE_ID"] == "OCCULTATION":
        if ir_sampling == "NORMAL":
            ir_fov = oops.fov.Flat(IR_NORMAL_SCALE, uv_shape, uv_los)
        else:
            ir_fov = oops.fov.Flat(IR_HIRES_SCALE, uv_shape, uv_los)

    elif ir_sampling in ("HI-RES","UNDER"):
        ir_fov = oops.fov.Flat(IR_HIRES_SCALE, uv_shape,
                    (IR_HIRES_FACTOR * uv_los[0] - uv_shape[0]/2., uv_los[1]))

    elif uv_shape == (64,64):
        ir_fov = IR_FULL_FOV

    else:
        ir_fov = oops.fov.Flat(IR_NORMAL_SCALE, uv_shape, uv_los)

    if ir_sampling == "UNDER":
        ir_det_size = 2.
    else:
        ir_det_size = 1.

    ########################################
    # Define the cadences
    ########################################

    # Define cadences based on header parameters
    ir_texp  = info["EXPOSURE_DURATION"][0] * 0.001
    vis_texp = info["EXPOSURE_DURATION"][1] * 0.001

    interframe_delay = info["INTERFRAME_DELAY_DURATION"] * 0.001
    interline_delay  = info["INTERLINE_DELAY_DURATION"]  * 0.001

    length_stride = max(ir_texp * swath_width, vis_texp) + interline_delay
    frame_stride = swath_length * length_stride + (interframe_delay -
                                                  interline_delay)

    frame_size = swath_width * swath_length
    frames = (samples * lines) / frame_size
    assert samples * lines == frames * frame_size

    frame_cadence = None
    backplane_cadence = None

    # Define a cadence based on the time backplane, if it is present
    if times is None:
        pass

    elif info["OVERWRITTEN_CHANNEL_FLAG"] == "ON":
        times = times.ravel()
        assert times[0] < times[1]
        assert vis_is_off
        backplane_cadence = oops.cadence.Sequence(times, ir_texp)

    elif info["PACKING"] == "ON":
        times = times.ravel()
        assert times[0] == times[1]
        tstart = times[0]
        frame_cadence = oops.cadence.Sequence(times[::frame_size], texp=0.)

    else:       # No packing plus no embedded timing just means a better tstart
        times = times.ravel()
        assert times[0] == times[1]
        tstart = times[0]

    vis_header_cadence = oops.cadence.Metronome(tstart,
                                length_stride, vis_texp, swath_length)
    ir_fast_cadence = oops.cadence.Metronome(tstart,
                                ir_texp, ir_texp, swath_width)
    ir_header_cadence = oops.cadence.DualCadence(vis_header_cadence,
                                ir_fast_cadence)

    if frames > 1:
        frame_cadence = oops.cadence.Metronome(tstart,
                                frame_stride, frame_stride, frames)

    # At this point...
    #   vis_header_cadence  always defined, always 1-D.
    #   ir_fast_cadence     always defined, always 1-D.
    #   ir_header_cadence   always defined, always 2-D.
    #   backplane_cadence   defined if timing was recorded, always 1-D.
    #   frame_cadence       defined if multiple objects are packed into a single
    #                       file, always 1-D.

    ########################################
    # Define the coordinate frames
    ########################################

    vis_frame_id = "CASSINI_VIMS_V"
    ir_frame_id  = "CASSINI_VIMS_IR"

    if (info["TARGET_NAME"] == "SUN" or "_SOL" in info["OBSERVATION_ID"]):
        ir_frame_id = "CASSINI_VIMS_IR_SOL"

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
            fastcad = oops.cadence.Metronome(tstart, fast_stride, ir_texp,
                                             samples)

            slow_stride = ir_texp * samples + interline_delay
            slowcad = oops.cadence.Metronome(tstart, slow_stride, slow_stride,
                                             lines)
            fullcad = oops.cadence.DualCadence(slowcad, fastcad)
            ir_cadence = oops.cadence.ReshapedCadence(fullcad, (samples*lines,))

        else:
            ir_cadence = backplane_cadence

        ir_data = ir_data.reshape((frames, 256))
        ir_obs = oops.obs.Pixel(("t","b"),
                                ir_cadence, ir_fov,
                                "CASSINI", ir_frame_id, dict=label)

    # Single LINE case
    elif swath_length == 1 and frames == 1:
        if not vis_is_off:
            if vis_data is not None: vis_data = vis_data.reshape((samples, 96))

            vis_obs = oops.obs.Slit1D(("u","b"), 1.,
                                tstart, vis_texp, vis_fov,
                                "CASSINI", vis_frame_id, dict=label)

        if not ir_is_off:
            if ir_data is not None: ir_data = ir_data.reshape((samples, 256))

            if backplane_cadence is not None:
                ir_fast_cadence = backplane_cadence

            ir_obs = oops.obs.RasterSlit1D(("ut","b"), ir_det_size,
                                ir_fast_cadence, ir_fov,
                                "CASSINI", ir_frame_id, dict=label)

    # Single 2-D IMAGE case
    elif samples == swath_width and lines == swath_length:
        if not vis_is_off:
            vis_obs = oops.obs.Pushbroom(("vt","u","b"), (1.,1.),
                                vis_header_cadence, vis_fov,
                                "CASSINI", vis_frame_id, dict=label)

        if not ir_is_off:
            if backplane_cadence is None:
                ir_cadence = oops.cadence.DualCadence(vis_header_cadence,
                                ir_fast_cadence)
            else:
                ir_cadence = oops.cadence.ReshapedCadence(backplane_cadence,
                                (lines,samples))

            ir_obs = oops.obs.RasterScan(("vslow","ufast","b"),
                                (1., ir_det_size),
                                ir_cadence, ir_fov,
                                "CASSINI", ir_frame_id, dict=label)

    # Multiple LINE case
    elif swath_length == 1 and swath_length == lines:
        if not vis_is_off:
            vis_obs = oops.obs.Slit(("vt","u","b"), 1.,
                                frame_cadence, vis_fov,
                                "CASSINI", vis_frame_id, dict=label)

        if not ir_is_off:
            if backplane_cadence is None:
                ir_cadence = oops.cadence.DualCadence(frame_cadence,
                                ir_fast_cadence)
            else:
                ir_cadence = oops.cadence.ReshapedCadence(backplane_cadence,
                                (lines,samples))

            ir_obs = oops.obs.RasterSlit(("vslow","ufast","b"), ir_det_size,
                                ir_cadence, ir_fov,
                                "CASSINI", ir_frame_id, dict=label)

    # Multiple 2-D IMAGE case
    elif lines == frames and samples == swath_width * swath_length:
        if not vis_is_off:

            # Reshape the data array
            if vis_data is not None:
                vis_data = vis_data.reshape(frames, swath_length, swath_width,
                                vis_data.shape[-1])

            # Define the first 2-D pushbroom observation
            vis_first_obs = oops.obs.Pushbroom(("t", "vt","u","b"), (1.,1.),
                                vis_header_cadence, vis_fov,
                                "CASSINI", vis_frame_id)

            # Define the movie
            movie_cadence = oops.cadence.DualCadence(frame_cadence,
                                vis_header_cadence)

            vis_obs = oops.obs.Movie(("t","vt","u","b"), vis_first_obs,
                                movie_cadence, dict=label)

        if not vis_is_off:

            # Reshape the data array
            if ir_data is not None:
                ir_data = ir_data.reshape(frames, swath_length, swath_width,
                                          ir_data.shape[-1])

            # Define the first 2-D raster-scan observation
            ir_first_obs = oops.obs.RasterScan(("vslow","ufast","b"),
                                (1., ir_det_size),
                                ir_first_cadence, ir_fov,
                                "CASSINI", ir_frame_id)

            # Define the 3-D cadence
            if backplane_cadence is None:
                ir_first_cadence = oops.cadence.DualCadence(vis_header_cadence,
                                ir_fast_cadence)

                ir_cadence = oops.cadence.DualCadence(frame_cadence,
                                ir_first_cadence)
            else:
                ir_cadence = oops.cadence.ReshapedCadence(backplane_cadence,
                                (frames,lines,samples))

            # Define the movie
            ir_obs = oops.obs.Movie(("t","vslow","ufast","b"), ir_first_obs,
                                ir_cadence, dict=label)

    else:
        raise ValueError("unsupported VIMS format in file " + filespec)

    # Insert the data array
    if vis_obs is not None and vis_data is not None:
        vis_obs.insert_subfield("data", vis_data)

    if ir_obs is not None and ir_data is not None:
        ir_obs.insert_subfield("data", ir_data)

    return (vis_obs, ir_obs)

########################################

def _load_data_and_times(filespec, label):
    """Loads the data array from the file. If time backplanes are present, it
    also returns an array of times in seconds TDB as derived from these
    backplanes.

    Input:
        filespec        full path to the data file.
        label           the label dictionary.

    Return:             (data, times)
        data            a Numpy array containing the data in axis order
                        (line, sample, band).
        times           the time sampling array in (line, sample) axis order, or
                        None if no time backplane is found in the file.

    Note: This procedure is absurdly complicated but it has been rather
    carefully debugged. --MRS 7/4/12.
    """

    info = label["QUBE"]

    # Extract key parameters fro the file header
    core_items   = info["CORE_ITEMS"]
    core_samples = core_items[0]
    core_bands   = core_items[1]
    core_lines   = core_items[2]
    core_item_bytes = info["CORE_ITEM_BYTES"]
    core_item_type  = info["CORE_ITEM_TYPE"]

    sample_suffix_items = info["SUFFIX_ITEMS"][0]
    band_suffix_items   = info["SUFFIX_ITEMS"][1]

    suffix_item_bytes = 4

    if sample_suffix_items == 1:
        suffix_item_type = info["SAMPLE_SUFFIX_ITEM_TYPE"]
    elif sample_suffix_items > 1:
        suffix_item_type = info["SAMPLE_SUFFIX_ITEM_TYPE"][0]
    elif band_suffix_items == 1:
        suffix_item_type = info["BAND_SUFFIX_ITEM_TYPE"]
    elif band_suffix_items > 1:
        suffix_item_type = info["BAND_SUFFIX_ITEM_TYPE"][0]
    else:
        suffix_item_type = ""

    record_bytes = label["RECORD_BYTES"]
    header_bytes = record_bytes * (label["^QUBE"] - 1)

    # Make sure we have byte-aligned values
    assert (core_samples * core_item_bytes) % suffix_item_bytes == 0

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
    offset = header_bytes / core_item_bytes
    size = line_stride * core_lines

    # Determine the dtype for the file core
    if "SUN_" in core_item_type or "MSB_" in core_item_type:
        core_dtype = ">"
    elif "PC_" in core_item_type or  "LSB_" in core_item_type:
        core_dtype = "<"
    else:
        raise TypeError("Unrecognized byte order: " + core_item_type)

    if "UNSIGNED" in core_item_type:
        core_dtype += "u"
        native_dtype = "int"
    elif "INTEGER" in core_item_type:
        core_dtype += "i"
        native_dtype = "int"
    elif "REAL"    in core_item_type:
        core_dtype += "f"
        native_dtype = "float"
    else:
        raise TypeError("Unrecognized core data type: " + core_item_type)

    core_dtype += str(core_item_bytes)

    # Read the file as core dtypes
    array = np.fromfile(filespec, dtype=core_dtype)

    # Slice away the core lines, leaving off the line suffix
    array = array[offset:offset+size]

    # Create a data array using new strides in (line, sample, band) order
    data = numpy.lib.stride_tricks.as_strided(array,
                    strides = (line_stride, core_item_bytes, band_stride),
                    shape   = (core_lines,  core_samples,    core_bands))

    # Convert core to a native 3-D array
    data = data.astype(native_dtype)

    # If there are no time backplanes, we're done
    band_suffix_name = info["BAND_SUFFIX_NAME"]
    if "SLICE_TIME_SECONDS" not in band_suffix_name:
        return (data, None)

    ############################

    # Determine the dtype for the file core
    if "SUN_" in core_item_type or "MSB_" in suffix_item_type:
        suffix_item_dtype = ">"
    elif "PC_" in core_item_type or  "LSB_" in suffix_item_type:
        suffix_item_dtype = "<"
    else:
        raise TypeError("Unrecognized byte order: " + suffix_item_type)

    if "UNSIGNED" in suffix_item_type:
        suffix_item_dtype += "u"
        native_dtype = "int"
    elif "INTEGER" in suffix_item_type:
        suffix_item_dtype += "i"
        native_dtype = "int"
    elif "REAL"    in suffix_item_type:
        suffix_item_dtype += "f"
        native_dtype = "float"
    else:
        raise TypeError("Unrecognized suffix data type: " + suffix_item_type)

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
    backplane = numpy.lib.stride_tricks.as_strided(offset_array,
                strides = (backplane_stride,  line_stride, suffix_item_bytes),
                shape   = (band_suffix_items, core_lines,  core_samples))

    # Convert to spacecraft clock
    seconds = backplane[band_suffix_name.index("SLICE_TIME_SECONDS")]
    ticks = backplane[band_suffix_name.index("SLICE_TIME_TICKS")]
    sclock = seconds + ticks/15959.

    # Convert to TDB
    mask = (seconds == -8192)       # Sometimes all are -8192 except first
    if np.any(mask):
        assert np.all(mask.ravel()[1:])
        formatted = "%16.3f" % sclock[0,0]
        times = np.empty(sclock.shape)
        times[...] = cspice.scs2e(-82, formatted)

    else:
        sclock_min = sclock.min()
        formatted = "%16.3f" % sclock_min
        tdb_min = cspice.scs2e(-82, formatted) + (sclock_min - float(formatted))

        sclock_max = sclock.max()
        formatted = "%16.3f" % sclock_max
        tdb_max = cspice.scs2e(-82, formatted) + (sclock_max - float(formatted))

        times = tdb_min + (sclock - sclock_min) * ((tdb_max - tdb_min) /
                                                   (sclock_max - sclock_min))

    return (data, times)

########################################

# def from_index(filespec, parameters={}):
#     """A static method to return a list of Snapshot objects, one for each row
#     in an VIMS index file. The filespec refers to the label of the index file.
#     """
# 
#     def _vims_repair_line(line):
#         if "KM/SECOND" not in line:
#             return line.replace("KM/", "KM\"")
#         return line
# 
#     def _is_vims_comment(line):
#         if "DESCRIPTION             =" in line:
#             return True
#         elif "=" in line:
#             return False
#         elif "END" == line:
#             return False
#         return True
# 
#     def _vims_from_index(filespec):
#         core_lines = pdsparser.PdsLabel.load_file(filespec)
# 
#         # Deal with corrupt syntax
#         newlines = []
#         for line in core_lines:
#             if not _is_vims_comment(line):
#                 newlines.append(_vims_repair_line(line))
# 
#         table = pdstable.PdsTable(filespec, ["START_TIME", "STOP_TIME"],
#                                   newlines)
#         return table
# 
#     VIMS.initialize()    # Define everything the first time through
# 
#     # Read the index file
#     table = _vims_from_index(filespec)
#     row_qubes = table.dicts_by_row()
# 
#     # Create a list of Snapshot objects
#     observations = []
#     for dict in row_dicts:
#         time = dict["START_TIME"].value
#         if time[-1] == "Z": time = time[:-1]
#         tdb0 = cspice.str2et(time)
# 
#         time = dict["STOP_TIME"].value
#         if time[-1] == "Z": time = time[:-1]
#         tdb1 = cspice.str2et(time)
# 
#         inter_frame_delay = dict["INTERFRAME_DELAY_DURATION"] * 0.001
#         inter_line_delay = dict["INTERLINE_DELAY_DURATION"] * 0.001
#         swath_width = int(dict["SWATH_WIDTH"])
#         swath_length = int(dict["SWATH_LENGTH"])
#         x_offset = dict["X_OFFSET"]
#         z_offset = dict["Z_OFFSET"]
# 
#         exposure_duration = dict["EXPOSURE_DURATION"]
#         ir_exposure = exposure_duration[0] * 0.001
#         vis_exposure = exposure_duration[1] * 0.001
# 
#         total_row_time = inter_line_delay + max(ir_exposure * swath_width,
#                                                 vis_exposure)
# 
#         target_name = dict["TARGET_NAME"]
# 
#         # both the following two core_lines seem to produce the string "IMAGE"
#         instrument_mode = dict["INSTRUMENT_MODE_ID"]
#         instrument_mode_id = dict["INSTRUMENT_MODE_ID"]
# 
#         ir_pb = oops.obs.Pushbroom(0, total_row_time, swath_width * ir_exposure,
#                                    target_name, (tdb0, tdb1), VIMS.fovs["IR"],
#                                    "CASSINI", "CASSINI_VIMS_IR")
# 
#         vis_pb = oops.obs.Pushbroom(0, total_row_time, vis_exposure,
#                                     target_name, (tdb0, tdb1), VIMS.fovs["V"],
#                                     "CASSINI", "CASSINI_VIMS_V")
# 
#         observations.append((ir_pb, vis_pb))
# 
#     # Make sure all the SPICE kernels are loaded
#     tdb0 = row_dicts[0]["START_TIME"]
#     tdb1 = row_dicts[-1]["STOP_TIME"]
# 
#     Cassini.load_cks( tdb0, tdb1)
#     Cassini.load_spks(tdb0, tdb1)
# 
#     return observations
# 
# # Internal function to return the sum of elements as an int
# def _sumover(item):
#     try:
#         return sum(item)
#     except TypeError:
#         return int(item)

################################################################################

class VIMS(object):
    """A instance-free class to hold Cassini VIMS instrument parameters."""

    instrument_kernel = None
    fovs = {}
    initialized = False

    @staticmethod
    def initialize():
        """Fills in key information about the WAC and NAC. Must be called first.
        """

        # Quick exit after first call
        if VIMS.initialized: return

        Cassini.initialize()
        Cassini.load_instruments()

        # Load the instrument kernel
        VIMS.instrument_kernel = Cassini.spice_instrument_kernel("VIMS")[0]

        # Construct a flat FOV for each camera
        # Tuple is (SPK abbrev, my abbrev, hi-res factor)
        for det in {("V","VIS",3,3), ("IR","IR",2,1)}:
            info = VIMS.instrument_kernel["INS"]["CASSINI_VIMS_" + det[0]]

            # Full field of view
            core_lines = info["PIXEL_LINES"]
            core_samples = info["PIXEL_SAMPLES"]
            assert core_lines == 64
            assert core_samples == 64

            xfov = info["FOV_REF_ANGLE"]
            yfov = info["FOV_CROSS_ANGLE"]
            assert info["FOV_ANGLE_UNITS"] == "DEGREES"

            uscale = np.arctan(np.tan(xfov * oops.RPD) / 32.)
            vscale = np.arctan(np.tan(yfov * oops.RPD) / 32.)

            # Display directions: [u,v] = [right,down]
            normal_fov = oops.fov.Flat((uscale,vscale), (64,64))
            hires_fov  = oops.fov.Flat((uscale / det[2], vscale / det[3]),
                                       (64 * det[2], 64 * det[3]))

            # Load the dictionary
            VIMS.fovs[(det[1],"NORMAL")] = normal_fov
            VIMS.fovs[(det[1],"HI-RES")] = hires_fov
            VIMS.fovs[(det[1],"N/A")] = normal_fov      # just prevents KeyError

        # Create a usable Nyquist FOV for the IR channel
        VIMS.fovs[("IR","UNDER" )] = VIMS.fovs[("IR","HI-RES")]

        # Construct a SpiceFrame for each detector
        ignore = oops.frame.SpiceFrame("CASSINI_VIMS_V")
        ignore = oops.frame.SpiceFrame("CASSINI_VIMS_IR")
        ignore = oops.frame.SpiceFrame("CASSINI_VIMS_IR_SOL")

        VIMS.initialized = True

    @staticmethod
    def reset():
        """Resets the internal Cassini VIMS parameters. Can be useful for
        debugging."""

        VIMS.instrument_kernel = None
        VIMS.fovs = {}
        VIMS.initialized = False

        Cassini.reset()

################################################################################
# Initialize at load time
################################################################################

VIMS.initialize()

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Cassini_VIMS(unittest.TestCase):

    def runTest(self):

        pass

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################


