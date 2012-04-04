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

# Function to return the sum of elements as an int
def sumover(item):
    try:
        return sum(item)
    except TypeError:
        return int(item)

def create_vis_observation(dict, filespec=None, label=None):
    # Get key information from the header
    if filespec is None:
        tdb0 = dict["START_TIME"]
        tdb1 = dict["STOP_TIME"]
    else:
        time = dict["START_TIME"].value
        if time[-1] == "Z": time = time[:-1]
        tdb0 = cspice.str2et(time)
    
        time = dict["STOP_TIME"].value
        if time[-1] == "Z": time = time[:-1]
        tdb1 = cspice.str2et(time)
        
        Cassini.load_cks( tdb0, tdb1)
        Cassini.load_spks(tdb0, tdb1)

    inter_frame_delay_msec = dict["INTERFRAME_DELAY_DURATION"]
    inter_line_delay_msec = dict["INTERLINE_DELAY_DURATION"]
    inter_frame_delay = inter_frame_delay_msec * .001
    inter_line_delay = inter_line_delay_msec * .001
    
    total_time = tdb1 - tdb0
    
    swath_width = int(dict["SWATH_WIDTH"])
    swath_length = int(dict["SWATH_LENGTH"])
    
    # what do we do with these offsets?
    x_offset = dict["X_OFFSET"]
    z_offset = dict["Z_OFFSET"]
    
    exposure_duration = dict["EXPOSURE_DURATION"]
    ir_exposure = exposure_duration[0] * .001
    vis_exposure = exposure_duration[1] * .001

    texp = max(ir_exposure * swath_width, vis_exposure)
    total_row_time = texp + inter_line_delay
    
    # not sure how to use the offset to the start of the visual exposure
    offset_to_vis_start = (ir_exposure * swath_width - vis_exposure) * 0.5
    
    target_name = dict["TARGET_NAME"]
    
    time_tuple = (tdb0, tdb1)
    
    # both the following two lines seem to produce the string "IMAGE"
    # not particularly useful for determining the FOV
    instrument_mode = dict["INSTRUMENT_MODE_ID"]
    instrument_mode_id = dict["INSTRUMENT_MODE_ID"]
    
    ir_fov = VIMS.fovs["IR"]
    ir_pb = oops.obs.Pushbroom(0, total_row_time, swath_width * ir_exposure,
                               target_name, time_tuple, ir_fov, "CASSINI",
                               "CASSINI_VIMS_IR")
    
    vis_fov = VIMS.fovs["V"]
    vis_pb = oops.obs.Pushbroom(0, total_row_time, vis_exposure,
                                target_name, time_tuple, vis_fov,
                                "CASSINI", "CASSINI_VIMS_V")
    
    # create data if we are getting from_file
    if filespec is not None:
        
        core_samples = int(dict["CORE_ITEMS"][0])
        core_bands   = int(dict["CORE_ITEMS"][1])
        core_lines   = int(dict["CORE_ITEMS"][2])
        
        core_bytes   = int(dict["CORE_ITEM_BYTES"])
        
        suffix_samples = int(dict["SUFFIX_ITEMS"][0])
        suffix_bands   = int(dict["SUFFIX_ITEMS"][1])
        suffix_lines   = int(dict["SUFFIX_ITEMS"][2])
        
        if suffix_samples > 0:
            sample_suffix_bytes = sumover(dict["SAMPLE_SUFFIX_ITEM_BYTES"])
        else:
            sample_suffix_bytes = 0
        
        if suffix_bands > 0:
            band_suffix_bytes = sumover(dict["BAND_SUFFIX_ITEM_BYTES"])
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
        #pylab.imshow(slice)
        #raw_input("press enter to continue:")
        
        vis_cube = cube[0:96,l,:]
        ir_cube = cube[96:256,l,:]
    
        ir_pb.insert_subfield("data", ir_cube)
        vis_pb.insert_subfield("data", vis_cube)
    
    return (ir_pb, vis_pb)


def from_file(filespec, parameters={}):
    """A general, static method to return an Snapshot object based on a given
        Cassini VIMS image file."""
    
    VIMS.initialize()    # Define everything the first time through
    
    # Load the VICAR file
    label = pdsparser.PdsLabel.from_file(filespec)
    qube = label["QUBE"]
    
    return create_vis_observation(qube, filespec, label)

################################################################################

def vims_process_line(line):
    if "KM/SECOND" not in line:
        return line.replace("KM/", "KM\"")
    return line
    
def is_vims_comment(line):
    if "DESCRIPTION             =" in line:
        return True
    elif "=" in line:
        return False
    elif "END" == line:
        return False
    return True

def vims_from_index(filespec):
    lines = pdsparser.PdsLabel.load_file(filespec)
    
    # Deal with corrupt syntax
    newlines = []
    for line in lines:
        if not is_vims_comment(line):
            newlines.append(vims_process_line(line))
    
    table = pdstable.PdsTable(filespec, ["START_TIME", "STOP_TIME"], newlines)
    return table

def from_index(filespec, parameters={}):
    """A static method to return a list of Snapshot objects, one for each row
        in an VIMS index file. The filespec refers to the label of the index file.
        """
    
    VIMS.initialize()    # Define everything the first time through
    
    # Read the index file
    table = vims_from_index(filespec)
    row_dicts = table.dicts_by_row()
    
    # Create a list of Snapshot objects
    observations = []
    for dict in row_dicts:
        observations.append(create_vis_observation(dict))

    
    # Make sure all the SPICE kernels are loaded
    tdb0 = row_dicts[0]["START_TIME"]
    tdb1 = row_dicts[-1]["STOP_TIME"]
    
    Cassini.load_cks( tdb0, tdb1)
    Cassini.load_spks(tdb0, tdb1)
    
    return observations


################################################################################

class VIMS(object):
    """A instance-free class to hold Cassini VIMS instrument parameters."""
    
    instrument_kernel = None
    fovs = {}
    initialized = False
    
    @staticmethod
    def initialize():
        """Fills in key information about the WAC and NAC. Must be called first."""
        
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
            
            uscale = np.arctan(np.tan(xfov * np.pi/180.) / (samples/2.))
            vscale = np.arctan(np.tan(yfov * np.pi/180.) / (lines/2.))
            
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
        #ignore = oops.frame.SpiceFrame("CASSINI_VIMS_RAD",
        #                               id="CASSINI_VIMS_RAD_FLIPPED")
        ignore = oops.frame.SpiceFrame("CASSINI_VIMS_V",
                                       id="CASSINI_VIMS_V_FLIPPED")
        
        rot180 = oops.Matrix3([[-1,0,0],[0,-1,0],[0,0,1]])
        ignore = oops.frame.Cmatrix(rot180, "CASSINI_VIMS_IR_SOL",
                                    "CASSINI_VIMS_IR_SOL_FLIPPED")
        ignore = oops.frame.Cmatrix(rot180, "CASSINI_VIMS_IR",
                                    "CASSINI_VIMS_IR_FLIPPED")
        #ignore = oops.frame.Cmatrix(rot180, "CASSINI_VIMS_RAD",
        #                            "CASSINI_VIMS_RAD_FLIPPED")
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
        #from_index("test_data/cassini/VIMS/index.lbl")
        """obs = from_index("test_data/cassini/VIMS/COVIMS_0016/INDEX.LBL")
        for ob in obs:
            print "observation time:"
            print ob[0].time"""

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################


