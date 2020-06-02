################################################################################
# oops/inst/cassini/vims.py
#
# 7/24/12 MRS -- First working version
# 1/9/15 MRS -- All delta-times multiplied by 1.01725 to correct error reported
#   in document "VIMS IR Pixel Timing_final.pdf". Special case of packing
#   multiple frames with swath_length = 1 as one frame with multiple lines.
#
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
import cspyce
import oops

from oops.inst.cassini.cassini_ import Cassini

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

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, fast=False, **parameters):
    """A general, static method to return a pair of Observation objects based on
    a given Cassini VIMS data file or label file.

    Input:
        filespec        the full path to a VIMS cube file or its PDS label.
        fast            True to perform a fast load of a label file. A fast load
                        bypasses the PDS label parser and pulls the minimum
                        required information out of the label using a much
                        faster procedure. It works only when the filespec is a
                        PDS label.

    Return:             (vis,ir)
        vis             the VIS observation, or None if the VIS channel was
                        inactive.
        ir              the IR observation, or None if the IR channel was
                        inactive.
    """

    VIMS.initialize()   # Define everything the first time through; use defaults
                        # unless initialize() is called explicitly.

    # Load a label via the fast procedure if specified
    if fast:
        assert filespec.lower().endswith(".lbl")
        f = open(filespec)
        lines = f.readlines()
        f.close()

        label = fast_dict(lines)

        # Allow label["SPECTRAL_QUBE"] to work properly below
        label["SPECTRAL_QUBE"] = label

    # Otherwise, use the standard parser
    else:
        # Load the VIMS file or the PDS label
        lines = pdsparser.PdsLabel.load_file(filespec)

        # Fix known syntax errors
        for i in range(len(lines)):
            line = lines[i]

            # In GAIN_MODE_ID and BACKGROUND_SAMPLING_MODE_ID, sometimes N/A is
            # not properly quoted
            if line[:4] in ("GAIN", "BACK"):
                lines[i] = line.replace('(N/A',  '("N/A"')
                lines[i] = line.replace('N/A)',  '"N/A")')

            # Sometimes a comment begins on one line and and ends on the next
            if line.strip().endswith("*/") and "/*" not in line:
                lines[i] = "\n"

        label = pdsparser.PdsLabel.from_string(lines).as_dict()

    is_isis_file = "QUBE" in label.keys()
    if is_isis_file:                # If this is an ISIS file...
        info = label["QUBE"]        # ... the info is in the QUBE object
    else:                           # Otherwise, this is a .LBL file
        info = label                # ... and the info is at the top level
        info["CORE_ITEMS"] = label["SPECTRAL_QUBE"]["CORE_ITEMS"]
        info["BAND_SUFFIX_NAME"] = label["SPECTRAL_QUBE"]["BAND_SUFFIX_NAME"]
        info["PACKING"] = info["PACKING_FLAG"]

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
    assert bands in (352, 256, 96)

    if is_isis_file:
        (data, times) = _load_data_and_times(filespec, label)
        assert data.shape == (lines, samples, bands)

        if bands == 352:
            vis_data = data[:,:,:96]    # index order is [line, sample, band]
            ir_data  = data[:,:,96:]
        elif bands == 256:              # only happens in a few early cubes
            vis_data = None
            ir_data = data
        else:
            vis_data = data
            ir_data = None

    else:
        vis_data = None
        ir_data = None
        times = None

    ########################################
    # Define the FOVs
    ########################################

    swath_width = info["SWATH_WIDTH"]
    swath_length = info["SWATH_LENGTH"]

    frame_size = swath_width * swath_length
    frames = (samples * lines) / frame_size
    assert samples * lines == frames * frame_size

    # Replace multiple one-line frames by one frame, multiple lines
    if frames > 1 and frame_size != 1:
        assert swath_length == 1
        swath_length = frames
        lines = frames
        frame_size = swath_width * swath_length

    uv_shape = (swath_width, swath_length)

    x_offset = info["X_OFFSET"]
    z_offset = info["Z_OFFSET"]
    uv_los = (33. - x_offset, 33. - z_offset)

    vis_sampling = info["SAMPLING_MODE_ID"][1].strip()  # handle " NORMAL"
    ir_sampling  = info["SAMPLING_MODE_ID"][0].strip()

    # VIS FOV
    if vis_sampling == "HI-RES":
        vis_fov = oops.fov.FlatFOV(VIS_HIRES_SCALE, uv_shape,
                    (VIS_HIRES_FACTOR * uv_los[0] - uv_shape[0],
                     VIS_HIRES_FACTOR * uv_los[1] - uv_shape[1]))

    elif uv_shape == (64,64):
        vis_fov = VIS_FULL_FOV

    else:
        vis_fov = oops.fov.FlatFOV(VIS_NORMAL_SCALE, uv_shape,
                    (IR_OVER_VIS * uv_los[0], IR_OVER_VIS * uv_los[1]))

    # IR FOV
    if info["INSTRUMENT_MODE_ID"] == "OCCULTATION":
        if ir_sampling == "NORMAL":
            ir_fov = oops.fov.FlatFOV(IR_NORMAL_SCALE, uv_shape, uv_los)
        else:
            ir_fov = oops.fov.FlatFOV(IR_HIRES_SCALE, uv_shape, uv_los)

    elif ir_sampling in ("HI-RES","UNDER"):
        ir_fov = oops.fov.FlatFOV(IR_HIRES_SCALE, uv_shape,
                    (IR_HIRES_FACTOR * uv_los[0] - uv_shape[0]/2., uv_los[1]))

    elif uv_shape == (64,64):
        ir_fov = IR_FULL_FOV

    else:
        ir_fov = oops.fov.FlatFOV(IR_NORMAL_SCALE, uv_shape, uv_los)

    if ir_sampling == "UNDER":
        ir_det_size = 2.
    else:
        ir_det_size = 1.

    ########################################
    # Define the cadences
    ########################################

    # Define cadences based on header parameters
    ir_texp  = info["EXPOSURE_DURATION"][0] * 0.001 * TIME_FACTOR
    vis_texp = info["EXPOSURE_DURATION"][1] * 0.001 * TIME_FACTOR
    vis_texp_nonzero = max(vis_texp, 1.e-8) # avoids divide-by-zero in cadences

    interframe_delay = info["INTERFRAME_DELAY_DURATION"] * 0.001 * TIME_FACTOR
    interline_delay  = info["INTERLINE_DELAY_DURATION"]  * 0.001 * TIME_FACTOR

    # Adjust the timing of one line, multiple frames
    if frames > 1 and frame_size != 1:
        interline_delay = interframe_delay

    length_stride = max(ir_texp * swath_width, vis_texp) + interline_delay

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
                                length_stride, vis_texp_nonzero, swath_length)
    ir_fast_cadence = oops.cadence.Metronome(tstart,
                                ir_texp, ir_texp, swath_width)
    ir_header_cadence = oops.cadence.DualCadence(vis_header_cadence,
                                ir_fast_cadence)

    # At this point...
    #   vis_header_cadence  always defined, always 1-D.
    #   ir_fast_cadence     always defined, always 1-D.
    #   ir_header_cadence   always defined, always 2-D.
    #   backplane_cadence   defined if timing was recorded, always 1-D.

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

        if ir_data is not None:
            ir_data = ir_data.reshape((frames, 256))

        ir_obs = oops.obs.Pixel(("t","b"),
                                ir_cadence, ir_fov,
                                "CASSINI", ir_frame_id)

    # Single LINE case
    elif swath_length == 1:
        if not vis_is_off:
            if vis_data is not None: vis_data = vis_data.reshape((samples, 96))

            vis_obs = oops.obs.Slit1D(("u","b"), 1.,
                                tstart, vis_texp_nonzero, vis_fov,
                                "CASSINI", vis_frame_id)

        if not ir_is_off:
            if ir_data is not None: ir_data = ir_data.reshape((samples, 256))

            if backplane_cadence is not None:
                ir_fast_cadence = backplane_cadence

            ir_obs = oops.obs.RasterSlit1D(("ut","b"), ir_det_size,
                                ir_fast_cadence, ir_fov,
                                "CASSINI", ir_frame_id)

    # Single 2-D IMAGE case
    elif samples == swath_width and lines == swath_length:
        if not vis_is_off:
            vis_obs = oops.obs.Pushbroom(("vt","u","b"), (1.,1.),
                                vis_header_cadence, vis_fov,
                                "CASSINI", vis_frame_id)

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
                                "CASSINI", ir_frame_id)

    # Multiple LINE case
    elif swath_length == 1 and swath_length == lines:
        if not vis_is_off:
            vis_obs = oops.obs.Slit(("vt","u","b"), 1.,
                                frame_cadence, vis_fov,
                                "CASSINI", vis_frame_id)

        if not ir_is_off:
            if backplane_cadence is None:
                ir_cadence = oops.cadence.DualCadence(frame_cadence,
                                ir_fast_cadence)
            else:
                ir_cadence = oops.cadence.ReshapedCadence(backplane_cadence,
                                (lines,samples))

            ir_obs = oops.obs.RasterSlit(("vslow","ufast","b"), ir_det_size,
                                ir_cadence, ir_fov,
                                "CASSINI", ir_frame_id)

# 1/9/15 broken code no longer needed
#     # Multiple 2-D IMAGE case
#     elif lines == frames and samples == swath_width * swath_length:
#         if not vis_is_off:
# 
#             # Reshape the data array
#             if vis_data is not None:
#                 vis_data = vis_data.reshape(frames, swath_length, swath_width,
#                                 vis_data.shape[-1])
# 
#             # Define the first 2-D pushbroom observation
#             vis_first_obs = oops.obs.Pushbroom(("t", "vt","u","b"), (1.,1.),
#                                 vis_header_cadence, vis_fov,
#                                 "CASSINI", vis_frame_id)
# 
#             # Define the movie
#             movie_cadence = oops.cadence.DualCadence(frame_cadence,
#                                 vis_header_cadence)
# 
#             vis_obs = oops.obs.Movie(("t","vt","u","b"), vis_first_obs,
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
#             ir_first_obs = oops.obs.RasterScan(("vslow","ufast","b"),
#                                 (1., ir_det_size),
#                                 ir_first_cadence, ir_fov,
#                                 "CASSINI", ir_frame_id)
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
#             ir_obs = oops.obs.Movie(("t","vslow","ufast","b"), ir_first_obs,
#                                 ir_cadence)
# 
#             # Reshape the data array
#             if ir_data is not None:
#                 ir_data = ir_data.reshape(frames, swath_length, swath_width,
#                                           ir_data.shape[-1])
# 
#             # Define the first 2-D raster-scan observation
#             ir_first_obs = oops.obs.RasterScan(("vslow","ufast","b"),
#                                 (1., ir_det_size),
#                                 ir_first_cadence, ir_fov,
#                                 "CASSINI", ir_frame_id)
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
#             ir_obs = oops.obs.Movie(("t","vslow","ufast","b"), ir_first_obs,
#                                 ir_cadence)

    else:
        raise ValueError("unsupported VIMS format in file " + filespec)

    # Insert the data array
    if vis_obs is not None:
        vis_obs.insert_subfield("instrument", "VIMS")
        vis_obs.insert_subfield("detector", "VIS")
        vis_obs.insert_subfield("sampling", vis_sampling)
        vis_obs.insert_subfield("dict", label)
        vis_obs.insert_subfield("index_dict", label)# for backward compatibility

        if vis_data is not None:
            vis_obs.insert_subfield("data", vis_data)

    if ir_obs is not None:
        ir_obs.insert_subfield("instrument", "VIMS")
        ir_obs.insert_subfield("detector", "IR")
        ir_obs.insert_subfield("sampling", ir_sampling)
        ir_obs.insert_subfield("dict", label)
        ir_obs.insert_subfield("index_dict", label)# for backward compatibility

        if ir_data is not None:
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
        times[...] = cspyce.scs2e(-82, formatted)

    else:
        sclock_min = sclock.min()
        formatted = "%16.3f" % sclock_min
        tdb_min = cspyce.scs2e(-82, formatted) + (sclock_min - float(formatted))

        sclock_max = sclock.max()
        formatted = "%16.3f" % sclock_max
        tdb_max = cspyce.scs2e(-82, formatted) + (sclock_max - float(formatted))

        times = tdb_min + (sclock - sclock_min) * ((tdb_max - tdb_min) /
                                                   (sclock_max - sclock_min))

    return (data, times)

########################################

def pds_value_from_constants(string_value):
    """Returns a value or tuple from a string that does not contain quotes and
        therefore expects a pre-defined constant
        
    Input:
        string_value    a string that is either a pre-defined constant OR a
                        tuple of pre-defined constants
        
    Return:             a string of that constant or a tuple of strings of those
                        constants.
    """
    if string_value[0] == '(' and string_value[-1] == ')':
        words = string_value[1:-1].split(',')
        return words
        """string_list = []
        for word in words:
            string_list.append('"' + word + '"')
        string_tuple = tuple(string_list)
        return string_tuple"""
    #if add_quotes:
    #    return '"' + string_value + '"'
    return string_value

def fast_dict(lines):
    """Returns a dictionary extracted from the PDS label of a VIMS file,
    containing the minimum required set of entries for the observation to be
    generated and analyzed. This routine is much faster than a call to
    PdsLabel.from_file(), because it does not use the pyparsing module.

    Input:
        lines           a list containing all the lines of the file, as read by
                        file.readlines().

    Return:             a dictionary containing, at minimum, these elements:
                            "BAND_SUFFIX_NAME" = unparsed string
                            "CORE_ITEMS" = three ints
                            "EXPOSURE_DURATION" = two floats
                            "INSTRUMENT_MODE_ID" = unquoted string
                            "INTERFRAME_DELAY_DURATION" = float
                            "INTERLINE_DELAY_DURATION" = float
                            "MISSION_PHASE_NAME" = string
                            "OVERWRITTEN_CHANNEL_FLAG" = string
                            "PACKING_FLAG" = string
                            "POWER_STATE_FLAG" = pair of strings
                            "SAMPLING_MODE_ID" = pair of strings
                            "START_TIME" = string
                            "SWATH_LENGTH" = int
                            "SWATH_WIDTH" = int
                            "TARGET_NAME" = string
                            "X_OFFSET" = int
                            "Z_OFFSET" = int
    """

    def three_ints(string):
        string = string.strip()
        assert string[0] == "(" and string[1] == ")"
        (a,b,c) = string[1:-1].split(",")
        return (int(a), int(b), int(c))

    def two_floats(string):
        string = string.strip()
        assert string[0] == "(" and string[1] == ")"
        (a,b) = string[1:-1].split(",")
        return (float(a), float(b))

    def two_strings(string):
        string = string.strip()
        assert string[0] == "(" and string[1] == ")"
        (a,b) = string[1:-1].split(",")
        return (unquote(a), unquote(b))

    def unquote(string):
        string = string.strip()
        if len(string) >= 2 and string[0] == '"' and string[-1] == '"':
            return string[1:-1]
        else:
            return string

    # Create a quick dictionary using the PDS keyword, equal to the string value
    dict = {}
    for line in lines:
        pair = line.split("=")
        if len(pair) == 2:
            dict[pair[0].strip()] = pair[1]

    # Re-define the required keywords

    dict["BAND_SUFFIX_NAME"]          = dict["BAND_SUFFIX_NAME"].strip()
    dict["CORE_ITEMS"]                = three_ints(dict["CORE_ITEMS"])
    dict["EXPOSURE_DURATION"]         = two_floats(dict["EXPOSURE_DURATION"])
    dict["INSTRUMENT_MODE_ID"]        = unquote(dict["INSTRUMENT_MODE_ID"])
    dict["INTERFRAME_DELAY_DURATION"] = float(dict["INTERFRAME_DELAY_DURATION"])
    dict["INTERLINE_DELAY_DURATION"]  = float(dict["INTERLINE_DELAY_DURATION"])
    dict["MISSION_PHASE_NAME"]        = unquote(["MISSION_PHASE_NAME"])
    dict["OVERWRITTEN_CHANNEL_FLAG"]  = unquote(["OVERWRITTEN_CHANNEL_FLAG"])
    dict["PACKING_FLAG"]              = unquote(dict["PACKING_FLAG"])
    dict["POWER_STATE_FLAG"]          = two_strings(dict["POWER_STATE_FLAG"])
    dict["SAMPLING_MODE_ID"]          = two_strings(dict["SAMPLING_MODE_ID"])
    dict["START_TIME"]                = dict["START_TIME"].strip()
    dict["SWATH_LENGTH"]              = int(dict["SWATH_LENGTH"])
    dict["SWATH_WIDTH"]               = int(dict["SWATH_WIDTH"])
    dict["TARGET_NAME"]               = unquote(dict["TARGET_NAME"])
    dict["X_OFFSET"]                  = int(dict["X_OFFSET"])
    dict["Z_OFFSET"]                  = int(dict["Z_OFFSET"])

    return dict

########################################

def meshgrid_and_times(obs, oversample=6, extend=1.5):
    """Returns a meshgrid object and time array that oversamples and extends the
    dimensions of the field of view of a VIMS observation.

    Input:
        obs             the VIMS observation object to for which to generate a
                        meshgrid and a time array.
        oversample      the factor by which to oversample the field of view, in
                        units of the full-resolution VIMS pixel size.
        extend          the number of pixels by which to extend the field of
                        view, in units of the oversampled pixel.

    Return:             (mesgrid, time)
    """

    shrinkage = {("IR",  "NORMAL"): (1,1),
                 ("IR",  "HI-RES"): (2,1),
                 ("IR",  "UNDER" ): (2,1),
                 ("VIS", "NORMAL"): (1,1),
                 ("VIS", "HI-RES"): (3,3)}

    assert obs.instrument == "VIMS"

    (ushrink,vshrink) = shrinkage[(obs.detector, obs.sampling)]

    oversample = float(oversample)
    undersample = (ushrink, vshrink)

    ustep = ushrink / oversample
    vstep = vshrink / oversample

    origin = (-extend * ustep, -extend * vstep)

    limit = (obs.fov.uv_shape.vals[0] + extend * ustep,
             obs.fov.uv_shape.vals[1] + extend * vstep)

    meshgrid = oops.Meshgrid.for_fov(obs.fov, origin, undersample, oversample,
                                     limit, swap=True)

    time = obs.uvt(obs.fov.nearest_uv(meshgrid.uv).swapxy())[1]

    return (meshgrid, time)

################################################################################

def initialize(ck='reconstructed', planets=None, asof=None,
               spk='reconstructed', gapfill=True,
               mst_pck=True, irregulars=True):
    """Initialize key information about the VIMS instrument.

    Must be called first. After the first call, later calls to this function
    are ignored.

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

    VIMS.initialize(ck=ck, planets=planets, asof=asof, spk=spk,
                    gapfill=gapfill, mst_pck=mst_pck, irregulars=irregulars)

################################################################################

class VIMS(object):
    """A instance-free class to hold Cassini VIMS instrument parameters."""

    initialized = False
    instrument_kernel = None

    @staticmethod
    def initialize(ck='reconstructed', planets=None, asof=None,
                   spk='reconstructed', gapfill=True,
                   mst_pck=True, irregulars=True):
        """Fills in key information about the VIS and IR channels.

        Must be called first. After the first call, later calls to this function
        are ignored.

        Input:
            ck,spk      'predicted', 'reconstructed', or 'none', depending on
                        which kernels are to be used. Defaults are
                        'reconstructed'. Use 'none' if the kernels are to be
                        managed manually.
            planets     A list of planets to pass to define_solar_system. None
                        or 0 means all.
            asof        Only use SPICE kernels that existed before this date;
                        None to ignore.
            gapfill     True to include gapfill CKs. False otherwise.
            mst_pck     True to include MST PCKs, which update the rotation
                        models for some of the small moons.
            irregulars  True to include the irregular satellites;
                        False otherwise.
        """

        # Quick exit after first call
        if VIMS.initialized: return

        Cassini.initialize(ck=ck, planets=planets, asof=asof, spk=spk,
                           gapfill=gapfill,
                           mst_pck=mst_pck, irregulars=irregulars)
        Cassini.load_instruments(asof=asof)

        # Load the instrument kernel
        VIMS.instrument_kernel = Cassini.spice_instrument_kernel("VIMS")[0]

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


