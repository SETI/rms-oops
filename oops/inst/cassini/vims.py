################################################################################
# oops/inst/cassini/iss.py
################################################################################

import numpy as np
import tempfile
import pylab
import julian
import pdstable
import pdsparser
import cspice
import oops

import utils as cassini

INSTRUMENT_KERNEL = None
FOVS = {}

################################################################################
# Standard class methods
################################################################################

# Function to return the sum of elements as an int
def sumover(item):
    try:
        return sum(item)
    except TypeError:
        return int(item)

def from_file(filespec, parameters={}):
    """A general, static method to return an Snapshot object based on a given
        Cassini ISS image file."""
    
    initialize()    # Define everything the first time through
    
    # Load the VICAR file
    label = pdsparser.PdsLabel.from_file(filespec)
    qube = label["QUBE"]
    #print "qube:"
    #print qube

    # Get key information from the header
    time = qube["START_TIME"].value
    if time[-1] == "Z": time = time[:-1]
    tdb0 = cspice.str2et(time)
    
    time = qube["STOP_TIME"].value
    if time[-1] == "Z": time = time[:-1]
    tdb1 = cspice.str2et(time)
    
    #midtime = (tdb0 + tdb1) * 0.5
    
    inter_frame_delay_msec = qube["INTERFRAME_DELAY_DURATION"]
    inter_line_delay_msec = qube["INTERLINE_DELAY_DURATION"]
    inter_frame_delay = inter_frame_delay_msec * .001
    inter_line_delay = inter_line_delay_msec * .001
    
    total_time = tdb1 - tdb0
    
    swath_width = int(qube["SWATH_WIDTH"])
    swath_length = int(qube["SWATH_LENGTH"])
    
    # what do we do with these offsets?
    x_offset = qube["X_OFFSET"]
    z_offset = qube["Z_OFFSET"]
    
    exposure_duration = qube["EXPOSURE_DURATION"]
    ir_exposure = exposure_duration[0] * .001
    vis_exposure = exposure_duration[1] * .001
    
    core_samples = int(qube["CORE_ITEMS"][0])
    core_bands   = int(qube["CORE_ITEMS"][1])
    core_lines   = int(qube["CORE_ITEMS"][2])
    
    core_bytes   = int(qube["CORE_ITEM_BYTES"])
    
    suffix_samples = int(qube["SUFFIX_ITEMS"][0])
    suffix_bands   = int(qube["SUFFIX_ITEMS"][1])
    suffix_lines   = int(qube["SUFFIX_ITEMS"][2])
    
    if suffix_samples > 0:
        sample_suffix_bytes = sumover(qube["SAMPLE_SUFFIX_ITEM_BYTES"])
    else:
        sample_suffix_bytes = 0
    
    if suffix_bands > 0:
        band_suffix_bytes = sumover(qube["BAND_SUFFIX_ITEM_BYTES"])
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
    core_type = str(qube["CORE_ITEM_TYPE"])
    
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
    # Identify empty regions
    flags = (str(qube["POWER_STATE_FLAG"][0]) != "OFF",
             str(qube["POWER_STATE_FLAG"][1]) != "OFF")

    native_start_time = float(qube["NATIVE_START_TIME"])
    native_stop_time = float(qube["NATIVE_STOP_TIME"])

    texp = max(ir_exposure * swath_width, vis_exposure)
    total_row_time = texp + inter_line_delay
    print "ir_exposure:"
    print ir_exposure
    print "vis_exposure:"
    print vis_exposure
    print "texp:"
    print texp
    
    # not sure how to use the offset to the start of the visual exposure
    offset_to_vis_start = (ir_exposure * swath_width - vis_exposure) * 0.5

    target_name = qube["TARGET_NAME"]

    time_tuple = (tdb0, tdb1)

    # both the following two lines seem to produce the string "IMAGE"
    # not particularly useful for determining the FOV
    instrument_mode = qube["INSTRUMENT_MODE_ID"]
    instrument_mode_id = qube["INSTRUMENT_MODE_ID"]
    
    ir_fov = FOVS["IR"]
    ir_pb = oops.obs.Pushbroom(0, total_row_time, swath_width * ir_exposure,
                               target_name, time_tuple, ir_fov, "CASSINI",
                               "CASSINI_VIMS_IR")
    ir_pb.insert_subfield("data", ir_cube)
    
    vis_fov = FOVS["V"]
    vis_pb = oops.obs.Pushbroom(0, total_row_time, vis_exposure,
                                target_name, time_tuple, vis_fov,
                                "CASSINI", "CASSINI_VIMS_V")
    vis_pb.insert_subfield("data", vis_cube)

    # statistics
    print "total time:"
    print total_time
    print "total_row_time:"
    print total_row_time
    print "inter_frame_delay:"
    print inter_frame_delay
    print "core_lines:"
    print core_lines
    print "calculated total time:"
    print total_row_time * core_lines + inter_frame_delay
    print "diff between total_time and calculated time:"
    print total_time - (total_row_time * core_lines + inter_frame_delay)

    return (ir_pb, vis_pb)

################################################################################

def vims_process_line(line):
    if "KM/SECOND" not in line:
        return line.replace("KM/", "KM\"")
    return line
    """if len(line) <= 5:
        return line
    if line[2] == ' ':
        return line
    if "=" not in line:
        if line[2] == ' ' and line[3] == ' ':
            return line
        newline = "  " + line
        print newline
        return newline

    return line"""
    
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
        #if "For full definitions of index fields" not in line:
            #newlines.append(vims_process_line(line))
    
    # there is no from_string() function for PdsTable, so write to temp file
    """f = tempfile.NamedTemporaryFile(delete=False)
    for line in newlines:
        f.write("%s\n" % line)
    f.close()
    # Get dictionary
    table = pdstable.PdsTable(f.name, ["START_TIME", "STOP_TIME"])"""
    f = open("/Users/bwells/lsrc/pds-tools/test.txt", "w")
    for line in newlines:
        f.write("%s\n" % line)
    f.close()
    table = pdstable.PdsTable(filespec, ["START_TIME", "STOP_TIME"], newlines)
    #table = pdsparser.PdsLabel.from_string(newlines)
    return table

def from_index(filespec, parameters={}):
    """A static method to return a list of Snapshot objects, one for each row
        in an ISS index file. The filespec refers to the label of the index file.
        """
    
    initialize()    # Define everything the first time through
    
    # Read the index file
    #table = pdstable.PdsTable(filespec, ["START_TIME", "STOP_TIME"])
    table = vims_from_index(filespec)
    row_dicts = table.dicts_by_row()
    
    # Create a list of Snapshot objects
    snapshots = []
    for dict in row_dicts:
        
        #print "dict:"
        #print dict
        for key in dict:
            print "%s = %s" % (key, dict[key])
        tdb0 = julian.tdb_from_tai(dict["START_TIME"])
        tdb1 = julian.tdb_from_tai(dict["STOP_TIME"])
        
        """inter_frame_delay_msec = dict["INTERFRAME_DELAY_DURATION"]
        inter_line_delay_msec = dict["INTERLINE_DELAY_DURATION"]
        core_lines   = int(dict["CORE_ITEMS"][2])
        inter_frame_delay = inter_frame_delay_msec * .001
        inter_line_delay = inter_line_delay_msec * .001
    
        exposure_duration = qube["EXPOSURE_DURATION"]
        ir_exposure = exposure_duration[0] * .001
        vis_exposure = exposure_duration[1] * .001

        total_time = tdb1 - tdb0
        print "total time:"
        print total_time
        print "total_row_time:"
        print total_row_time
        print "inter_frame_delay:"
        print inter_frame_delay
        print "core_lines:"
        print core_lines
        print "calculated total time:"
        print total_row_time * core_lines + inter_frame_delay
        print "diff between total_time and calculated time:"
        print total_time - (total_row_time * core_lines + inter_frame_delay)

        
        instrument_mode = qube["INSTRUMENT_MODE_ID"]
        instrument_mode_id = qube["INSTRUMENT_MODE_ID"]
        print "instrument_mode:"
        print instrument_mode
        print "instrument_mode_id:"
        print instrument_mode_id"""

    
    # Make sure all the SPICE kernels are loaded
    tdb0 = row_dicts[0]["START_TIME"]
    tdb1 = row_dicts[-1]["STOP_TIME"]
    
    cassini.load_cks( tdb0, tdb1)
    cassini.load_spks(tdb0, tdb1)
    
    #return snapshots


################################################################################

def initialize():
    """Fills in key information about the WAC and NAC. Must be called first."""
    
    global INSTRUMENT_KERNEL, FOVS
    
    cassini.load_instruments()
    
    # Quick exit after first call
    if INSTRUMENT_KERNEL is not None: return
    
    # Load the instrument kernel
    INSTRUMENT_KERNEL = cassini.spice_instrument_kernel("VIMS")[0]
    """print "\nINSTRUMENT_KERNEL:"
    for key in INSTRUMENT_KERNEL:
        if "INS" in key:
            for ins_key in INSTRUMENT_KERNEL[key]:
                print ins_key
                print INSTRUMENT_KERNEL[key][ins_key]
        else:
            print key
            print INSTRUMENT_KERNEL[key]"""
    #print INSTRUMENT_KERNEL
    
    # Construct a flat FOV for each camera
    #for detector in ["IR_SOL", "IR", "RAD", "V"]:
    for detector in ["IR_SOL", "IR", "V"]:
        info = INSTRUMENT_KERNEL["INS"]["CASSINI_VIMS_" + detector]
        
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
        
        # Load the dictionary, include the subsampling modes
        #FOVS[detector, "FULL"] = full_fov
        #FOVS[detector, "SUM2"] = oops.fov.Subsampled(full_fov, 2)
        #FOVS[detector, "SUM4"] = oops.fov.Subsampled(full_fov, 4)
        FOVS[detector] = full_fov
    
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

################################################################################
# UNIT TESTS
################################################################################

import unittest


class Test_Cassini_VIMS(unittest.TestCase):

    def runTest(self):
        
        # Create the pushbroom objects
        #pbs = from_file("test_data/cassini/VIMS/V1546355125_1.QUB")

        #from_file("test_data/cassini/VIMS/V1546355804_1.QUB")
        #from_index("test_data/cassini/VIMS/index.lbl")
        from_index("test_data/cassini/VIMS/COVIMS_0016/INDEX.LBL")

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################


