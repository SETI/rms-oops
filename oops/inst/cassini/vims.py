################################################################################
# oops/inst/cassini/vims.py
################################################################################

import numpy as np
import tempfile
import pylab
import julian
import pdstable
import pdsparser
import cspice
import oops

from oops.inst.cassini.cassini_ import Cassini

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, parameters={}):
    """A general, static method to return an Snapshot object based on a given
    Cassini VIMS image file."""

    VIMS.initialize()   # Define everything the first time through

    # Load the VICAR file
    label = pdsparser.PdsLabel.from_file(filespec)
    dict = label["QUBE"]

    # Get key information from the header
    time = dict["START_TIME"].value
    if time[-1] == "Z": time = time[:-1]
    tdb0 = cspice.str2et(time)

    time = dict["STOP_TIME"].value
    if time[-1] == "Z": time = time[:-1]
    tdb1 = cspice.str2et(time)

    Cassini.load_cks( tdb0, tdb1)
    Cassini.load_spks(tdb0, tdb1)

    inter_frame_delay = dict["INTERFRAME_DELAY_DURATION"].value * 0.001
    inter_line_delay = dict["INTERLINE_DELAY_DURATION"].value * 0.001
    swath_width = int(dict["SWATH_WIDTH"].value)
    swath_length = int(dict["SWATH_LENGTH"].value)
    x_offset = dict["X_OFFSET"].value
    z_offset = dict["Z_OFFSET"].value

    exposure_duration = dict["EXPOSURE_DURATION"]
    ir_exposure = exposure_duration[0].value * 0.001
    vis_exposure = exposure_duration[1].value * 0.001

    total_row_time = inter_line_delay + max(ir_exposure * swath_width,
                                            vis_exposure)

    target_name = dict["TARGET_NAME"].value

    # both the following two lines seem to produce the string "IMAGE"
    instrument_mode = dict["INSTRUMENT_MODE_ID"]
    instrument_mode_id = dict["INSTRUMENT_MODE_ID"]

#     ir_pb = oops.obs.Pushbroom(0, total_row_time, swath_width * ir_exposure,
#                                target_name, (tdb0, tdb1), VIMS.fovs["IR"],
#                                "CASSINI", "CASSINI_VIMS_IR")
# 
#     vis_pb = oops.obs.Pushbroom(0, total_row_time, vis_exposure,
#                                 target_name, (tdb0, tdb1), VIMS.fovs["V"],
#                                 "CASSINI", "CASSINI_VIMS_V")

    ir_pb = oops.obs.Snapshot((tdb0, tdb1), VIMS.fovs["IR"],
                               "CASSINI", "CASSINI_VIMS_IR", label=label)

    vis_pb = oops.obs.Snapshot((tdb0, tdb1), VIMS.fovs["V"],
                                "CASSINI", "CASSINI_VIMS_V", label=label)

    core_samples   = int(dict["CORE_ITEMS"][0])
    core_bands     = int(dict["CORE_ITEMS"][1])
    core_lines     = int(dict["CORE_ITEMS"][2])
    core_bytes     = int(dict["CORE_ITEM_BYTES"])
    suffix_samples = int(dict["SUFFIX_ITEMS"][0])
    suffix_bands   = int(dict["SUFFIX_ITEMS"][1])
    suffix_lines   = int(dict["SUFFIX_ITEMS"][2])

    if suffix_samples > 0:
        sample_suffix_bytes = _sumover(dict["SAMPLE_SUFFIX_ITEM_BYTES"])
    else:
        sample_suffix_bytes = 0

    if suffix_bands > 0:
        band_suffix_bytes = _sumover(dict["BAND_SUFFIX_ITEM_BYTES"])
    else:
        band_suffix_bytes = 0

    # Suffix samples in units of the core item
    suffix_samples_scaled = sample_suffix_bytes / core_bytes
    suffix_bands_scaled   = band_suffix_bytes / core_bytes

    stride = np.empty((3), "int")
    stride[0] = 1
    stride[1] = core_samples + suffix_samples_scaled
    stride[2] = (stride[1] * core_bands +
                 suffix_bands_scaled * (core_samples + suffix_samples))

    # Determine the dtype for the file core
    core_type = str(dict["CORE_ITEM_TYPE"])

    if "SUN_" in core_type or "MSB_" in core_type:
        core_dtype = ">"
    elif "PC_" in core_type or  "LSB_" in core_type:
        core_dtype = "<"
    else:
        raise TypeError("Unrecognized byte order: " + core_type)

    if  "UNSIGNED" in core_type: core_dtype += "u"
    elif "INTEGER" in core_type: core_dtype += "i"
    elif "REAL"    in core_type: core_dtype += "f"
    else:
        raise TypeError("Unrecognized data type: " + core_type)

    core_dtype += str(core_bytes)

    # Read the file
    buffer = np.fromfile(filespec, core_dtype)

    # Select the core items
    record_bytes = int(label["RECORD_BYTES"])
    qube_record  = int(label["^QUBE"])

    offset = record_bytes * (qube_record-1) / core_bytes
    size = stride[2] * core_lines

    buffer = buffer[offset:offset+size]

    # Organize by lines
    buffer = buffer.reshape(core_lines, stride[2])

    # Extract the core as a native 3-D array
    core_dtype = "=" + core_dtype[1:]
    cube = np.empty((core_bands, core_lines, core_samples), core_dtype)

    shape = (core_bands, stride[1])
    size  = shape[0] * shape[1]

    for l in range(core_lines):
        slice = buffer[l,0:size]
        slice = slice.reshape(shape)
        slice = slice[:,0:core_samples]

        cube[:,l,:] = slice[:,:]

    vis_cube = cube[0:96,:,:]
    ir_cube = cube[96:256,:,:]

    ir_pb.insert_subfield("data", ir_cube)
    vis_pb.insert_subfield("data", vis_cube)

    return (ir_pb, vis_pb)

def from_index(filespec, parameters={}):
    """A static method to return a list of Snapshot objects, one for each row
    in an VIMS index file. The filespec refers to the label of the index file.
    """

    def _vims_repair_line(line):
        if "KM/SECOND" not in line:
            return line.replace("KM/", "KM\"")
        return line

    def _is_vims_comment(line):
        if "DESCRIPTION             =" in line:
            return True
        elif "=" in line:
            return False
        elif "END" == line:
            return False
        return True

    def _vims_from_index(filespec):
        lines = pdsparser.PdsLabel.load_file(filespec)

        # Deal with corrupt syntax
        newlines = []
        for line in lines:
            if not _is_vims_comment(line):
                newlines.append(_vims_repair_line(line))

        table = pdstable.PdsTable(filespec, ["START_TIME", "STOP_TIME"],
                                  newlines)
        return table

    VIMS.initialize()    # Define everything the first time through

    # Read the index file
    table = _vims_from_index(filespec)
    row_dicts = table.dicts_by_row()

    # Create a list of Snapshot objects
    observations = []
    for dict in row_dicts:
        time = dict["START_TIME"].value
        if time[-1] == "Z": time = time[:-1]
        tdb0 = cspice.str2et(time)

        time = dict["STOP_TIME"].value
        if time[-1] == "Z": time = time[:-1]
        tdb1 = cspice.str2et(time)

        inter_frame_delay = dict["INTERFRAME_DELAY_DURATION"] * 0.001
        inter_line_delay = dict["INTERLINE_DELAY_DURATION"] * 0.001
        swath_width = int(dict["SWATH_WIDTH"])
        swath_length = int(dict["SWATH_LENGTH"])
        x_offset = dict["X_OFFSET"]
        z_offset = dict["Z_OFFSET"]

        exposure_duration = dict["EXPOSURE_DURATION"]
        ir_exposure = exposure_duration[0] * 0.001
        vis_exposure = exposure_duration[1] * 0.001

        total_row_time = inter_line_delay + max(ir_exposure * swath_width,
                                                vis_exposure)

        target_name = dict["TARGET_NAME"]

        # both the following two lines seem to produce the string "IMAGE"
        instrument_mode = dict["INSTRUMENT_MODE_ID"]
        instrument_mode_id = dict["INSTRUMENT_MODE_ID"]

        ir_pb = oops.obs.Pushbroom(0, total_row_time, swath_width * ir_exposure,
                                   target_name, (tdb0, tdb1), VIMS.fovs["IR"],
                                   "CASSINI", "CASSINI_VIMS_IR")

        vis_pb = oops.obs.Pushbroom(0, total_row_time, vis_exposure,
                                    target_name, (tdb0, tdb1), VIMS.fovs["V"],
                                    "CASSINI", "CASSINI_VIMS_V")

        observations.append((ir_pb, vis_pb))

    # Make sure all the SPICE kernels are loaded
    tdb0 = row_dicts[0]["START_TIME"]
    tdb1 = row_dicts[-1]["STOP_TIME"]

    Cassini.load_cks( tdb0, tdb1)
    Cassini.load_spks(tdb0, tdb1)

    return observations

# Internal function to return the sum of elements as an int
def _sumover(item):
    try:
        return sum(item)
    except TypeError:
        return int(item)

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
        #for detector in ["IR_SOL", "IR", "RAD", "V"]:
        for detector in ["IR_SOL", "IR", "V"]:
            info = VIMS.instrument_kernel["INS"]["CASSINI_VIMS_" + detector]

            # Full field of view
            lines = info["PIXEL_LINES"]
            samples = info["PIXEL_SAMPLES"]

            xfov = info["FOV_REF_ANGLE"]
            yfov = info["FOV_CROSS_ANGLE"]
            assert info["FOV_ANGLE_UNITS"] == "DEGREES"

            uscale = np.arctan(np.tan(xfov * oops.RPD) / (samples/2.))
            vscale = np.arctan(np.tan(yfov * oops.RPD) / (lines/2.))

            # Display directions: [u,v] = [right,down]
            full_fov = oops.fov.Flat((uscale,vscale), (samples,lines))

            # Load the dictionary
            VIMS.fovs[detector] = full_fov

        # Construct a SpiceFrame for each camera
        # Deal with the fact that the instrument's internal coordinate system is
        # rotated 180 degrees
        ignore = oops.frame.SpiceFrame("CASSINI_VIMS_IR_SOL",
                                       id="CASSINI_VIMS_IR_SOL_FLIPPED")
        ignore = oops.frame.SpiceFrame("CASSINI_VIMS_IR",
                                       id="CASSINI_VIMS_IR_FLIPPED")
        ignore = oops.frame.SpiceFrame("CASSINI_VIMS_V",
                                       id="CASSINI_VIMS_V_FLIPPED")

        rot180 = oops.Matrix3([[-1,0,0],[0,-1,0],[0,0,1]])
        ignore = oops.frame.Cmatrix(rot180, "CASSINI_VIMS_IR_SOL",
                                    "CASSINI_VIMS_IR_SOL_FLIPPED")
        ignore = oops.frame.Cmatrix(rot180, "CASSINI_VIMS_IR",
                                    "CASSINI_VIMS_IR_FLIPPED")
        ignore = oops.frame.Cmatrix(rot180, "CASSINI_VIMS_V",
                                    "CASSINI_VIMS_V_FLIPPED")
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

        # Create the pushbroom objects
        #pbs = from_file("test_data/cassini/VIMS/V1546355125_1.QUB")

        ob = from_file("test_data/cassini/VIMS/V1546355804_1.QUB")
        print "observation time:"
        print ob[0].time
        resolution = 8.0
        meshgrid = Meshgrid.for_fov(ob.fov, undersample=resolution, swap=True)
        bp = oops.Backplane(ob, meshgrid)


        #from_index("test_data/cassini/VIMS/index.lbl")
        """obs = from_index("test_data/cassini/VIMS/COVIMS_0016/INDEX.LBL")
        for ob in obs:
            print "observation time:"
            print ob[0].time"""

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################


