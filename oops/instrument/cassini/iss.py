################################################################################
# oops/instrument/cassini/iss.py
################################################################################

import numpy as np
import unittest

import julian
import vicar
import pdstable
import cspice

import oops
import oops.instrument

INSTRUMENT_KERNEL = None
FOVS = {}
DN_SCALING = None

import oops.instrument.cassini.utils as utils

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, parameters={}):
    """A general, static method to return an Snapshot object based on a given
    Cassini ISS image file."""

    initialize()    # Define everything the first time through

    # Load the VICAR file
    vic = vicar.VicarImage.from_file(filespec)

    # Get key information from the header
    time = vic["START_TIME"]
    if time[-1] == "Z": time = time[:-1]
    tdb0 = cspice.str2et(time)

    time = vic["STOP_TIME"]
    if time[-1] == "Z": time = time[:-1]
    tdb1 = cspice.str2et(time)

    mode = vic["INSTRUMENT_MODE_ID"]

    name = vic["INSTRUMENT_NAME"]
    if "WIDE" in name:
        camera = "WAC"
    else:
        camera = "NAC"

    # Make sure the SPICE kernels are loaded
    utils.load_cks( tdb0, tdb1)
    utils.load_spks(tdb0, tdb1)

    # Create a Snapshot
    result = oops.Snapshot(vic.get_2d_array(),      # data
                           None,                    # mask
                           ["v","u"],               # axes
                           (tdb0, tdb1),            # time
                           FOVS[camera,mode],       # fov
                           "CASSINI",               # path_id
                           "CASSINI_ISS_" + camera, # frame_id
                           DN_SCALING)              # calibration

    # Tack on the Vicar object in case more info is needed
    # This object behaves like a dictionary for most practical purposes
    result.dict = vic

    return result

################################################################################

def from_index(filespec, parameters={}):
    """A static method to return a list of Snapshot objects, one for each row
    in an ISS index file. The filespec refers to the label of the index file.
    """

    initialize()    # Define everything the first time through

    # Read the index file
    table = pdstable.PdsTable(filespec, ["START_TIME", "STOP_TIME"])
    row_dicts = table.dicts_by_row()

    # Create a list of Snapshot objects
    snapshots = []
    for dict in row_dicts:

        tdb0 = dict["START_TIME"]
        tdb1 = dict["STOP_TIME"]

        mode = dict["INSTRUMENT_MODE_ID"]

        name = dict["INSTRUMENT_NAME"]
        if "WIDE" in name:
            camera = "WAC"
        else:
            camera = "NAC"

        item = oops.Snapshot(None,                      # data
                             None,                      # mask
                             ["v","u"],                 # axes
                             (tdb0, tdb1),              # time
                             FOVS[camera,mode],         # fov
                             "CASSINI",                 # path_id
                             "CASSINI_ISS_" + camera,   # frame_id
                             DN_SCALING)                # calibration

        # Tack on the dictionary in case more info is needed
        item.dict = dict

        snapshots.append(item)

    # Make sure all the SPICE kernels are loaded
    tdb0 = row_dicts[0]["START_TIME"]
    tdb1 = row_dicts[-1]["STOP_TIME"]

    utils.load_cks( tdb0, tdb1)
    utils.load_spks(tdb0, tdb1)


    return snapshots

################################################################################

def initialize():
    """Fills in key information about the WAC and NAC. Must be called first."""

    global INSTRUMENT_KERNEL, FOVS, DN_SCALING

    utils.load_instruments()

    # Quick exit after first call
    if INSTRUMENT_KERNEL is not None: return

    # Load the instrument kernel
    INSTRUMENT_KERNEL = utils.spice_instrument_kernel("ISS")[0]

    # Construct a FlatFOV for each camera
    for detector in ["NAC", "WAC"]:
        info = INSTRUMENT_KERNEL["INS"]["CASSINI_ISS_" + detector]

        # Full field of view
        lines = info["PIXEL_LINES"]
        samples = info["PIXEL_SAMPLES"]

        xfov = info["FOV_REF_ANGLE"]
        yfov = info["FOV_CROSS_ANGLE"]
        assert info["FOV_ANGLE_UNITS"] == "DEGREES"

        uscale = np.arctan(np.tan(xfov * np.pi/180.) / (samples/2.))
        vscale = np.arctan(np.tan(yfov * np.pi/180.) / (lines/2.))

        # Display directions: [u,v] = [right,down]
        full_fov = oops.FlatFOV((uscale,vscale), (samples,lines))

        # Load the dictionary, include the subsampling modes
        FOVS[detector, "FULL"] = full_fov
        FOVS[detector, "SUM2"] = oops.SubsampledFOV(full_fov, 2)
        FOVS[detector, "SUM4"] = oops.SubsampledFOV(full_fov, 4)

    # Construct a SpiceFrame for each camera
    # Deal with the fact that the instrument's internal coordinate system is
    # rotated 180 degrees
    ignore = oops.SpiceFrame("CASSINI_ISS_NAC", id="CASSINI_ISS_NAC_FLIPPED")
    ignore = oops.SpiceFrame("CASSINI_ISS_WAC", id="CASSINI_ISS_WAC_FLIPPED")

    rot180 = oops.Matrix3([[-1,0,0],[0,-1,0],[0,0,1]])
    ignore = oops.MatrixFrame(rot180, "CASSINI_ISS_NAC_FLIPPED",
                                      "CASSINI_ISS_NAC")
    ignore = oops.MatrixFrame(rot180, "CASSINI_ISS_WAC_FLIPPED",
                                      "CASSINI_ISS_WAC")

    # Default scaling for raw images
    DN_SCALING = oops.Scaling("DN", 1.)

################################################################################
# UNIT TESTS
################################################################################

class Test_Cassini_ISS(unittest.TestCase):

    def runTest(self):

        snapshots = from_index("test_data/cassini/ISS/index.lbl")

        snapshot = from_file("test_data/cassini/ISS/W1575634136_1.IMG")
        vimg = vicar.VicarImage.from_file("test_data/cassini/ISS/W1575634136_1.IMG")
        self.assertTrue(np.all(snapshot.data == vimg.data[0]))
        ptable = pdstable.PdsTable("test_data/cassini/ISS/index.lbl")
        
        #print 'FRAME_REGISTRY'
        #print oops.FRAME_REGISTRY['SATURN_DESPUN']
        bp_data = snapshot.radius_back_plane()

        #test START_TIME for this file
        t0_col = ptable.column_dict['START_TIME']
        t0_str = t0_col[3940]
        (day,sec) = julian.day_sec_from_iso(t0_str)
        t0_tai = julian.tai_from_day(day) + sec
        t0_tdb = julian.tdb_from_tai(t0_tai)
        print "t0s:"
        print snapshot.t0
        print t0_tdb
        self.assertTrue(snapshot.t0 == t0_tdb)

        #test STOP_TIME for this file
        t1_col = ptable.column_dict['STOP_TIME']
        t1_str = t1_col[3940]
        (day,sec) = julian.day_sec_from_iso(t1_str)
        t1_tai = julian.tai_from_day(day) + sec
        t1_tdb = julian.tdb_from_tai(t1_tai)
        print "t1s:"
        print snapshot.t1
        print t1_tdb
        self.assertTrue(snapshot.t1 == t1_tdb)

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################


