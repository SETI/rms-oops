################################################################################
# oops/inst/cassini/iss.py
################################################################################

import numpy as np
import julian
import vicar
import pdstable
import oops

from oops.inst.cassini.cassini_ import Cassini

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, parameters={}):
    """A general, static method to return a Snapshot object based on a given
    Cassini ISS image file."""

    ISS.initialize()    # Define everything the first time through

    # Load the VICAR file
    vic = vicar.VicarImage.from_file(filespec)
    dict = vic.as_dict()

    # Get key information from the header
    tstart = julian.tdb_from_tai(julian.tai_from_iso(vic["START_TIME"]))
    texp = max(1.e-3, dict["EXPOSURE_DURATION"]) / 1000.
    mode = dict["INSTRUMENT_MODE_ID"]

    name = dict["INSTRUMENT_NAME"]
    if "WIDE" in name:
        camera = "WAC"
    else:
        camera = "NAC"

    # Make sure the SPICE kernels are loaded
    Cassini.load_cks( tstart, tstart + texp)
    Cassini.load_spks(tstart, tstart + texp)

    # Create a Snapshot
    result = oops.obs.Snapshot(("v","u"), tstart, texp, ISS.fovs[camera,mode],
                               "CASSINI", "CASSINI_ISS_" + camera,
                               dict = dict,               # Add the VICAR dict
                               data = vic.data_2d,      # Add the data array
                               instrument = "ISS",
                               detector = camera,
                               sampling = mode)

    return result

################################################################################

def from_index(filespec, parameters={}):
    """A static method to return a list of Snapshot objects, one for each row
    in an ISS index file. The filespec refers to the label of the index file.
    """

    ISS.initialize()    # Define everything the first time through

    # Read the index file
    COLUMNS = []        # Return all columns
    TIMES = ["START_TIME"]
    table = pdstable.PdsTable(filespec, columns=COLUMNS, times=TIMES)
    row_dicts = table.dicts_by_row()

    # Create a list of Snapshot objects
    snapshots = []
    for dict in row_dicts:

        tstart = julian.tdb_from_tai(dict["START_TIME"])
        texp = max(1.e-3, dict["EXPOSURE_DURATION"]) / 1000.
        mode = dict["INSTRUMENT_MODE_ID"]

        name = dict["INSTRUMENT_NAME"]
        if "WIDE" in name:
            camera = "WAC"
        else:
            camera = "NAC"

        item = oops.obs.Snapshot(("v","u"), tstart, texp, ISS.fovs[camera,mode],
                                 "CASSINI", "CASSINI_ISS_" + camera,
                                 dict = dict,       # Add index dictionary
                                 index_dict = dict, # Old name
                                 instrument = "ISS",
                                 detector = camera,
                                 sampling = mode)

        snapshots.append(item)

    # Make sure all the SPICE kernels are loaded
    tdb0 = row_dicts[ 0]["START_TIME"]
    tdb1 = row_dicts[-1]["START_TIME"]

    Cassini.load_cks( tdb0, tdb1)
    Cassini.load_spks(tdb0, tdb1)

    return snapshots

################################################################################

class ISS(object):
    """A instance-free class to hold Cassini ISS instrument parameters."""

    instrument_kernel = None
    fovs = {}
    initialized = False

    @staticmethod
    def initialize():
        """Fills in key information about the WAC and NAC. Must be called first.
        """

        # Quick exit after first call
        if ISS.initialized: return

        Cassini.initialize()
        Cassini.load_instruments()

        # Load the instrument kernel
        ISS.instrument_kernel = Cassini.spice_instrument_kernel("ISS")[0]

        # Construct a flat FOV for each camera
        for detector in ["NAC", "WAC"]:
            info = ISS.instrument_kernel["INS"]["CASSINI_ISS_" + detector]

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
            ISS.fovs[detector, "FULL"] = full_fov
            ISS.fovs[detector, "SUM2"] = oops.fov.Subsampled(full_fov, 2)
            ISS.fovs[detector, "SUM4"] = oops.fov.Subsampled(full_fov, 4)

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

        ISS.initialized = True

    @staticmethod
    def reset():
        """Resets the internal Cassini ISS parameters. Can be useful for
        debugging."""

        ISS.instrument_kernel = None
        ISS.fovs = {}
        ISS.initialized = False

        Cassini.reset()

################################################################################
# Initialize at load time
################################################################################

ISS.initialize()

################################################################################
# UNIT TESTS
################################################################################

import unittest
import os.path

class Test_Cassini_ISS(unittest.TestCase):

    def runTest(self):

        from oops.unittester_support    import TESTDATA_PARENT_DIRECTORY

        snapshots = from_index(os.path.join(TESTDATA_PARENT_DIRECTORY, "test_data/cassini/ISS/index.lbl"))
        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY, "test_data/cassini/ISS/W1575634136_1.IMG"))
        snapshot3940 = snapshots[3940]  #should be same as snapshot
    
        self.assertTrue(abs(snapshot.time[0] - snapshot3940.time[0]) < 1.e-3)
        self.assertTrue(abs(snapshot.time[1] - snapshot3940.time[1]) < 1.e-3)

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
