################################################################################
# oops/inst/cassini/iss.py
################################################################################

import numpy as np
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
    
    print "qube:"
    print qube
    
    # Get key information from the header
    time = qube["START_TIME"].value
    if time[-1] == "Z": time = time[:-1]
    tdb0 = cspice.str2et(time)
    
    time = qube["STOP_TIME"].value
    if time[-1] == "Z": time = time[:-1]
    tdb1 = cspice.str2et(time)
    
    #midtime = (tdb0 + tdb1) * 0.5
    
    print "\n\n\ntdb0:"
    print tdb0
    print "tdb1:"
    print tdb1
    
    inter_frame_delay = qube["INTERFRAME_DELAY_DURATION"]
    inter_line_delay = qube["INTERLINE_DELAY_DURATION"]
    
    print "inter_frame_delay:"
    print inter_frame_delay
    print "inter_line_delay:"
    print inter_line_delay
    
    print "print total time:"
    total_time = tdb1 - tdb0
    print total_time
    
    swath_width = int(qube["SWATH_WIDTH"])
    swath_length = int(qube["SWATH_LENGTH"])
    
    print "swath_width:"
    print swath_width
    print "swath_length:"
    print swath_length
    
    x_offset = qube["X_OFFSET"]
    z_offset = qube["Z_OFFSET"]
    
    print "x_offset:"
    print x_offset
    print "z_offset:"
    print z_offset
    
    exposure_duration = qube["EXPOSURE_DURATION"]
    ir_exposure = exposure_duration[0] * .001
    vis_exposure = exposure_duration[1] * .001
    print "exposure_duration:"
    print exposure_duration
    
    #print "number of lines based on time:"
    #print (tdb1 - tdb0) / inter_line_delay
    
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
    print "stride:"
    print stride
    
    # calculate what the time should be:
    print "core_samples:"
    print core_samples
    print "core_lines:"
    print core_lines
    print "suffix_samples:"
    print suffix_samples
    print "suffix_lines:"
    print suffix_lines
    print "core_bands:"
    print core_bands
    print "suffix_bands:"
    print suffix_bands
    
    band_suffix_name = qube["BAND_SUFFIX_NAME"]
    print "band_suffix_name:"
    print band_suffix_name
    
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
    
    total_time_bp = 0
    for l in range(core_lines):
        slice = buffer[l,0:size]
        slice = slice.reshape(shape)
        time_bp = slice[:,core_samples+1:shape[1]]
        slice_total_time = 0
        for tt in time_bp:
            if tt[0] > 0:
                slice_total_time += tt[0]
        total_time_bp += slice_total_time
        slice = slice[:,0:core_samples]
        
        cube[:,l,:] = slice[:,:]
    #pylab.imshow(slice)
    #raw_input("press enter to continue:")
    
    vis_cube = cube[0:96,l,:]
    ir_cube = cube[96:256,l,:]
    print "total_time_bp:"
    print total_time_bp
    # Identify empty regions
    flags = (str(qube["POWER_STATE_FLAG"][0]) != "OFF",
             str(qube["POWER_STATE_FLAG"][1]) != "OFF")
    
    native_start_time = float(qube["NATIVE_START_TIME"])
    native_stop_time = float(qube["NATIVE_STOP_TIME"])
    
    texp = max(ir_exposure * swath_width, vis_exposure)
    total_row_time = texp + inter_line_delay
    
    offset_to_vis_start = (ir_exposure * swath_width - vis_exposure) * 0.5
    print "offset_to_vis_start:"
    print offset_to_vis_start
    
    target_name = qube["TARGET_NAME"]
    
    time_tuple = (tdb0, tdb1)
    
    instrument_mode = qube["INSTRUMENT_MODE_ID"]
    print "instrument_mode:"
    print instrument_mode
    
    instrument_mode_id = qube["INSTRUMENT_MODE_ID"]
    print "instrument_mode_id:"
    print instrument_mode_id
    
    #print "qube:"
    #print qube
    
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
    
    return (ir_pb, vis_pb)
    
    """
        mode = vic["INSTRUMENT_MODE_ID"]
        
        name = vic["INSTRUMENT_NAME"]
        if "WIDE" in name:
        camera = "WAC"
        else:
        camera = "NAC"
        
        # Make sure the SPICE kernels are loaded
        cassini.load_cks( tdb0, tdb1)
        cassini.load_spks(tdb0, tdb1)
        
        
        # Create a Snapshot
        result = oops.obs.Snapshot((tdb0, tdb1),            # time
        FOVS[camera,mode],       # fov
        "CASSINI",               # path_id
        "CASSINI_ISS_" + camera) # frame_id
        
        # Insert the Vicar object as asubfield in case more info is needed.
        # This object behaves like a dictionary for most practical purposes.
        result.insert_subfield("vicar_dict", vic)
        result.insert_subfield("data", vic.get_2d_array())
        
        return result
        """

################################################################################

def uvis_from_index(filespec):
    lines = pdsparser.PdsLabel.load_file(filespec)
    
    # Deal with corrupt syntax
    newlines = []
    first_obj = "None"
    for line in lines:
        if "RECORD_TYPE" in line:
            #words = line.split("=")
            newline = "RECORD_TYPE             = FIXED_LENGTH"
            print "RECORD_TYPE line:"
            print newline
            newlines.append(newline)
        elif "^TABLE" in line:
            words = line.split("=")
            newlines.append("^INDEX_TABLE = " + words[-1])
            """newline = ""
            for word in words:
                sword = word.rstrip()
                if sword == "^TABLE":
                    newline = "^INDEX_TABLE"
                else:
                    newline += " " + sword
            newlines.append(newline)"""
        elif first_obj == "None":
            if "OBJECT" in line:
                words = line.split("=")
                first_obj = words[-1].strip()
            newlines.append(line)
        elif "END_OBJECT" in line and "=" not in line:
            newline = line + " = " + first_obj
            newlines.append(newline)
        else:
            newlines.append(line)
    
    f = open("/Users/bwells/lsrc/pds-tools/test.txt", "w")
    for line in newlines:
        f.write("%s\n" % line)
    f.close()
    table = pdstable.PdsTable(filespec, ["START_TIME", "STOP_TIME"], newlines)
    return table

def from_index(filespec, parameters={}):
    """A static method to return a list of Snapshot objects, one for each row
        in an ISS index file. The filespec refers to the label of the index file.
        """
    
    initialize()    # Define everything the first time through
    
    # Read the index file
    table = uvis_from_index(filespec)
    #table = pdstable.PdsTable(filespec, ["START_TIME", "STOP_TIME"])
    row_dicts = table.dicts_by_row()
    
    # Create a list of Snapshot objects
    snapshots = []
    for dict in row_dicts:
        for key in dict:
            print "%s = %s" % (key, dict[key])

        tdb0 = julian.tdb_from_tai(dict["START_TIME"])
        tdb1 = julian.tdb_from_tai(dict["STOP_TIME"])
        
    
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
    INSTRUMENT_KERNEL = cassini.spice_instrument_kernel("UVIS")[0]
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
    #for detector in ["SOL_OFF", "HSP", "FUV", "EUV", "SOLAR", "HDAC"]:
    for detector in ["SOL_OFF", "HSP", "FUV", "EUV", "SOLAR"]:
        info = INSTRUMENT_KERNEL["INS"]["CASSINI_UVIS_" + detector]
        
        # Full field of view
        lines = info["PIXEL_LINES"]
        samples = info["PIXEL_SAMPLES"]
        
        xfov = info["FOV_REF_ANGLE"]
        yfov = info["FOV_CROSS_ANGLE"]
        #assert info["FOV_ANGLE_UNITS"] == "DEGREES"
        conversion = 1.
        if info["FOV_ANGLE_UNITS"] == "DEGREES":
            conversion = np.pi/180.
        
        uscale = np.arctan(np.tan(xfov * conversion) / (samples/2.))
        vscale = np.arctan(np.tan(yfov * conversion) / (lines/2.))
        
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
    ignore = oops.frame.SpiceFrame("CASSINI_UVIS_SOL_OFF",
                                   id="CASSINI_UVIS_SOL_OFF_FLIPPED")
    ignore = oops.frame.SpiceFrame("CASSINI_UVIS_HSP",
                                   id="CASSINI_UVIS_HSP_FLIPPED")
    ignore = oops.frame.SpiceFrame("CASSINI_UVIS_FUV",
                                   id="CASSINI_UVIS_FUV_FLIPPED")
    ignore = oops.frame.SpiceFrame("CASSINI_UVIS_EUV",
                                   id="CASSINI_UVIS_EUV_FLIPPED")
    ignore = oops.frame.SpiceFrame("CASSINI_UVIS_SOLAR",
                                   id="CASSINI_UVIS_SOLAR_FLIPPED")
    #ignore = oops.frame.SpiceFrame("CASSINI_UVIS_HDAC",
    #                               id="CASSINI_UVIS_HDAC_FLIPPED")
    
    rot180 = oops.Matrix3([[-1,0,0],[0,-1,0],[0,0,1]])
    ignore = oops.frame.Cmatrix(rot180, "CASSINI_UVIS_SOL_OFF",
                                "CASSINI_UVIS_SOL_OFF_FLIPPED")
    ignore = oops.frame.Cmatrix(rot180, "CASSINI_UVIS_HSP",
                                "CASSINI_UVIS_HSP_FLIPPED")
    ignore = oops.frame.Cmatrix(rot180, "CASSINI_UVIS_FUV",
                                "CASSINI_UVIS_FUV_FLIPPED")
    ignore = oops.frame.Cmatrix(rot180, "CASSINI_UVIS_EUV",
                                "CASSINI_UVIS_EUV_FLIPPED")
    ignore = oops.frame.Cmatrix(rot180, "CASSINI_UVIS_SOLAR",
                                "CASSINI_UVIS_SOLAR_FLIPPED")
    #ignore = oops.frame.Cmatrix(rot180, "CASSINI_UVIS_HDAC",
    #                            "CASSINI_UVIS_HDAC_FLIPPED")

################################################################################
# UNIT TESTS
################################################################################

import unittest


class Test_Cassini_UVIS(unittest.TestCase):
    
    def runTest(self):

        #from_file("test_data/cassini/UVIS/EUV1999_007_17_05.DAT")
        #from_file("test_data/cassini/UVIS/EUV1999_007_17_05.LBL")
        from_index("test_data/cassini/UVIS/INDEX.LBL")

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################


