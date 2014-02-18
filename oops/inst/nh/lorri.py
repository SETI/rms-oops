################################################################################
# oops/inst/nh/lorri.py
################################################################################

import numpy as np
import julian
import pyfits
import pdstable
import oops

from oops.inst.nh.nh_ import NewHorizons

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, use_fits_geom=False, parameters={}):
    """A general, static method to return a Snapshot object based on a given
    NewHorizons LORRI image file."""

    LORRI.initialize()    # Define everything the first time through

    # Load the FITS file
    nh_file = pyfits.open(filespec)

    # Get key information from the header
    texp = nh_file[0].header["EXPTIME"]
    tdb_midtime = nh_file[0].header["SPCSCET"]
    tstart = tdb_midtime-texp/2
    binning_mode = nh_file[0].header["SFORMAT"]
    
    # Make sure the SPICE kernels are loaded
    NewHorizons.load_cks( tstart, tstart + texp)
    NewHorizons.load_spks(tstart, tstart + texp)

    path_id = "NEW HORIZONS"
    frame_id = "NH_LORRI_"+binning_mode
    data = nh_file[0].data
    
    if use_fits_geom:
        # We first need to construct a path from the Sun to NH
        vecx = -nh_file[0].header["SPCSSCX"]
        vecy = -nh_file[0].header["SPCSSCY"]
        vecz = -nh_file[0].header["SPCSSCZ"]
        ra = nh_file[0].header["SPCBRRA"]
        dec = nh_file[0].header["SPCBRDEC"]
        north_clk = nh_file[0].header["SPCEMEN"]
        scet = nh_file[0].header["SPCSCET"]
        path_id = "NH_LORRI_PATH_"+str(scet)
        frame_id = "NH_LORRI_FRAME_"+str(scet)
        sc_path = oops.path.FixedPath(oops.Vector3([vecx, vecy, vecz]), "SUN", "J2000", path_id)
        oops.frame.Cmatrix.from_ra_dec(ra, dec, north_clk, frame_id+"_FLIPPED", "J2000")
        # We have to flip the X/Y axes
        flipxy = oops.Matrix3([[0,1,0],
                               [1,0,0],
                               [0,0,1]])
        ignore = oops.frame.Cmatrix(flipxy, frame_id, frame_id+"_FLIPPED")
        
    # Create a Snapshot
    result = oops.obs.Snapshot(("v","u"), tstart, texp, LORRI.fovs[binning_mode],
                               path_id, frame_id,
                               data = data,      # Add the data array
                               instrument = "LORRI")
                               
    return result

################################################################################

def from_index(filespec, parameters={}):
    """A static method to return a list of Snapshot objects, one for each row
    in an LORRI index file. The filespec refers to the label of the index file.
    """

    assert False # XXX
    LORRI.initialize()    # Define everything the first time through

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

        item = oops.obs.Snapshot(("v","u"), tstart, texp, LORRI.fovs[camera,mode],
                                 "NEWHORIZONS", "NEWHORIZONS_LORRI_" + camera,
                                 dict = dict,       # Add index dictionary
                                 index_dict = dict, # Old name
                                 instrument = "LORRI",
                                 detector = camera,
                                 sampling = mode)

        snapshots.append(item)

    # Make sure all the SPICE kernels are loaded
    tdb0 = row_dicts[ 0]["START_TIME"]
    tdb1 = row_dicts[-1]["START_TIME"]

    NewHorizons.load_cks( tdb0, tdb1)
    NewHorizons.load_spks(tdb0, tdb1)

    return snapshots

################################################################################

class LORRI(object):
    """A instance-free class to hold NewHorizons LORRI instrument parameters."""

    instrument_kernel = None
    fovs = {}
    initialized = False

    @staticmethod
    def initialize():
        """Fills in key information about LORRI. Must be called first.
        """

        # Quick exit after first call
        if LORRI.initialized: return

        NewHorizons.initialize()
        NewHorizons.load_instruments()

        # Load the instrument kernel
        LORRI.instrument_kernel = NewHorizons.spice_instrument_kernel("LORRI")[0]

        # Construct a flat FOV for each camera
        for binning_mode, binning_id in [("1X1", -98301), ("4X4", -98302)]:
            info = LORRI.instrument_kernel["INS"][binning_id]

            # Full field of view
            lines = info["PIXEL_LINES"]
            samples = info["PIXEL_SAMPLES"]

            xfov = info["FOV_REF_ANGLE"]
            yfov = info["FOV_CROSS_ANGLE"]
            assert info["FOV_ANGLE_UNITS"] == "DEGREES"
            
            uscale = np.arctan(np.tan(xfov * np.pi/180.) / (samples/2.))
            vscale = np.arctan(np.tan(yfov * np.pi/180.) / (lines/2.))
            
            # Display directions: [u,v] = [right,down]
            full_fov = oops.fov.Flat((-uscale,-vscale), (samples,lines))

            # Load the dictionary
            LORRI.fovs[binning_mode] = full_fov

        # Construct a SpiceFrame for each camera
        ignore = oops.frame.SpiceFrame("NH_LORRI_1x1", id='NH_LORRI_1X1_FLIPPED')
        ignore = oops.frame.SpiceFrame("NH_LORRI_4x4", id='NH_LORRI_4X4_FLIPPED')

        # For some reason the SPICE IK gives the boresight along -Z instead of +Z,
        # so we have to flip all axes.
        flipxyz = oops.Matrix3([[-1, 0, 0],
                                [ 0,-1, 0],
                                [ 0, 0,-1]])
        oops.frame.Cmatrix(flipxyz, "NH_LORRI_1X1",
                           "NH_LORRI_1X1_FLIPPED")
        oops.frame.Cmatrix(flipxyz, "NH_LORRI_4X4",
                           "NH_LORRI_4X4_FLIPPED")

        LORRI.initialized = True

    @staticmethod
    def reset():
        """Resets the internal NewHorizons LORRI parameters. Can be useful for
        debugging."""

        LORRI.instrument_kernel = None
        LORRI.fovs = {}
        LORRI.initialized = False

        NewHorizons.reset()

################################################################################
# Initialize at load time
################################################################################

LORRI.initialize()

################################################################################
# UNIT TESTS
################################################################################

import unittest
import os.path

class Test_NewHorizons_LORRI(unittest.TestCase):

    def runTest(self):

        assert False
        
        from oops.unittester_support    import TESTDATA_PARENT_DIRECTORY

        snapshots = from_index(os.path.join(TESTDATA_PARENT_DIRECTORY, "nh/LORRI/index.lbl"))
        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY, "nh/LORRI/W1575634136_1.IMG"))
        snapshot3940 = snapshots[3940]  #should be same as snapshot
    
        self.assertTrue(abs(snapshot.time[0] - snapshot3940.time[0]) < 1.e-3)
        self.assertTrue(abs(snapshot.time[1] - snapshot3940.time[1]) < 1.e-3)

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
