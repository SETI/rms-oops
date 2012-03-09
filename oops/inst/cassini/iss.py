################################################################################
# oops/inst/cassini/iss.py
################################################################################

import numpy as np
import julian
import vicar
import pdstable
import cspice
import oops

import utils as cassini

INSTRUMENT_KERNEL = None
FOVS = {}
DN_SCALING = None

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
    cassini.load_cks( tdb0, tdb1)
    cassini.load_spks(tdb0, tdb1)

        
    # Create a Snapshot
    result = oops.obs.Snapshot(vic.get_2d_array(),      # data
                               ["v","u"],               # axes
                               (tdb0, tdb1),            # time
                               FOVS[camera,mode],       # fov
                               "CASSINI",               # path_id
                               "CASSINI_ISS_" + camera) # frame_id

    # Insert the Vicar object as asubfield in case more info is needed.
    # This object behaves like a dictionary for most practical purposes.
    result.insert_subfield("vicar_dict", vic)

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

        tdb0 = julian.tdb_from_tai(dict["START_TIME"])
        tdb1 = julian.tdb_from_tai(dict["STOP_TIME"])

        mode = dict["INSTRUMENT_MODE_ID"]

        name = dict["INSTRUMENT_NAME"]
        if "WIDE" in name:
            camera = "WAC"
        else:
            camera = "NAC"

        item = oops.obs.Snapshot(None,                      # data
                                 ["v","u"],                 # axes
                                 (tdb0, tdb1),              # time
                                 FOVS[camera,mode],         # fov
                                 "CASSINI",                 # path_id
                                 "CASSINI_ISS_" + camera)   # frame_id

        # Tack on the dictionary in case more info is needed
        item.insert_subfield("index_dict", dict)

        snapshots.append(item)

    # Make sure all the SPICE kernels are loaded
    tdb0 = row_dicts[0]["START_TIME"]
    tdb1 = row_dicts[-1]["STOP_TIME"]

    cassini.load_cks( tdb0, tdb1)
    cassini.load_spks(tdb0, tdb1)

    return snapshots

################################################################################

def initialize():
    """Fills in key information about the WAC and NAC. Must be called first."""

    global INSTRUMENT_KERNEL, FOVS, DN_SCALING

    cassini.load_instruments()

    # Quick exit after first call
    if INSTRUMENT_KERNEL is not None: return

    # Load the instrument kernel
    INSTRUMENT_KERNEL = cassini.spice_instrument_kernel("ISS")[0]

    # Construct a flat FOV for each camera
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
        full_fov = oops.fov.Flat((uscale,vscale), (samples,lines))

        # Load the dictionary, include the subsampling modes
        FOVS[detector, "FULL"] = full_fov
        FOVS[detector, "SUM2"] = oops.fov.Subsampled(full_fov, 2)
        FOVS[detector, "SUM4"] = oops.fov.Subsampled(full_fov, 4)

    # Construct a SpiceFrame for each camera
    # Deal with the fact that the instrument's internal coordinate system is
    # rotated 180 degrees
    ignore = oops.frame.SpiceFrame("CASSINI_ISS_NAC",
                                   id="CASSINI_ISS_NAC_FLIPPED")
    ignore = oops.frame.SpiceFrame("CASSINI_ISS_WAC",
                                   id="CASSINI_ISS_WAC_FLIPPED")

    rot180 = oops.Matrix3([[-1,0,0],[0,-1,0],[0,0,1]])
    ignore = oops.frame.Cmatrix(rot180, "CASSINI_ISS_NAC",
                                        "CASSINI_ISS_NAC_FLIPPED")
    ignore = oops.frame.Cmatrix(rot180, "CASSINI_ISS_WAC",
                                        "CASSINI_ISS_WAC_FLIPPED")

    # Default scaling for raw images
    DN_SCALING = oops.calib.Scaling("DN", 1.)

################################################################################
# UNIT TESTS
################################################################################

import unittest


class Test_Cassini_ISS(unittest.TestCase):

    def runTest(self):

        snapshots = from_index("test_data/cassini/ISS/index.lbl")
        snapshot = from_file("test_data/cassini/ISS/W1575634136_1.IMG")
        snapshot3940 = snapshots[3940]  #should be same as snapshot
    
        self.assertTrue(abs(snapshot.t0 - snapshot3940.t0) < 1.e-3)
        self.assertTrue(abs(snapshot.t1 - snapshot3940.t1) < 1.e-3)

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################


