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

def from_file(filespec, fast_distortion=True, **parameters):
    """A general, static method to return a Snapshot object based on a given
    Cassini ISS image file.
    
    fast_distortion is True to use a pre-inverted polynomial, False to use
    a dynamically solved polynomial, and None is use a FlatFOV."""

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

    filter1, filter2 = dict["FILTER_NAME"]
    
    gain_mode = None
    
    if dict["GAIN_MODE_ID"][:3] == "215":
        gain_mode = 0
    elif dict["GAIN_MODE_ID"][:2] == "95":
        gain_mode = 1
    elif dict["GAIN_MODE_ID"][:2] == "29":
        gain_mode = 2
    elif dict["GAIN_MODE_ID"][:2] == "12":
        gain_mode = 3

    # Make sure the SPICE kernels are loaded
    Cassini.load_cks( tstart, tstart + texp)
    Cassini.load_spks(tstart, tstart + texp)

    # Create a Snapshot
    result = oops.obs.Snapshot(("v","u"), tstart, texp, ISS.fovs[camera,mode,fast_distortion],
                               "CASSINI", "CASSINI_ISS_" + camera,
                               dict = dict,               # Add the VICAR dict
                               data = vic.data_2d,      # Add the data array
                               instrument = "ISS",
                               detector = camera,
                               sampling = mode,
                               filter1 = filter1,
                               filter2 = filter2,
                               gain_mode = gain_mode)

    return result

################################################################################

def from_index(filespec, **parameters):
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

        item = oops.obs.Snapshot(("v","u"), tstart, texp,
                                 ISS.fovs[camera,mode,False],
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

    # Create a master version of the NAC and WAC distortion models from
    #   Owen Jr., W.M., 2003. Cassini ISS Geometric Calibration of April 2003.
    #   JPL IOM 312.E-2003.
    # These polynomials convert from X,Y (radians) to U,V (pixels).
    
    NAC_F = 2002.703    # mm
    NAC_E2 = 8.28e-6    # / mm2
    NAC_E5 = 5.45e-6    # / mm
    NAC_E6 = -19.67e-6  # / mm
    NAC_KX = 83.33333   # samples/mm
    NAC_KY = 83.3428    # lines/mm
    
    NAC_COEFF = np.zeros((4,4,2))
    NAC_COEFF[1,0,0] = NAC_KX        * NAC_F
    NAC_COEFF[3,0,0] = NAC_KX*NAC_E2 * NAC_F**3
    NAC_COEFF[1,2,0] = NAC_KX*NAC_E2 * NAC_F**3
    NAC_COEFF[1,1,0] = NAC_KX*NAC_E5 * NAC_F**2
    NAC_COEFF[2,0,0] = NAC_KX*NAC_E6 * NAC_F**2
    
    NAC_COEFF[0,1,1] = NAC_KY        * NAC_F
    NAC_COEFF[2,1,1] = NAC_KY*NAC_E2 * NAC_F**3
    NAC_COEFF[0,3,1] = NAC_KY*NAC_E2 * NAC_F**3
    NAC_COEFF[0,2,1] = NAC_KY*NAC_E5 * NAC_F**2
    NAC_COEFF[1,1,1] = NAC_KY*NAC_E6 * NAC_F**2

    WAC_F = 200.7761    # mm
    WAC_E2 = 60.89e-6    # / mm2
    WAC_E5 = 4.93e-6    # / mm
    WAC_E6 = -72.28e-6  # / mm
    WAC_KX = 83.33333   # samples/mm
    WAC_KY = 83.34114   # lines/mm

    WAC_COEFF = np.zeros((4,4,2))
    WAC_COEFF[1,0,0] = WAC_KX        * WAC_F
    WAC_COEFF[3,0,0] = WAC_KX*WAC_E2 * WAC_F**3
    WAC_COEFF[1,2,0] = WAC_KX*WAC_E2 * WAC_F**3
    WAC_COEFF[1,1,0] = WAC_KX*WAC_E5 * WAC_F**2
    WAC_COEFF[2,0,0] = WAC_KX*WAC_E6 * WAC_F**2
    
    WAC_COEFF[0,1,1] = WAC_KY        * WAC_F
    WAC_COEFF[2,1,1] = WAC_KY*WAC_E2 * WAC_F**3
    WAC_COEFF[0,3,1] = WAC_KY*WAC_E2 * WAC_F**3
    WAC_COEFF[0,2,1] = WAC_KY*WAC_E5 * WAC_F**2
    WAC_COEFF[1,1,1] = WAC_KY*WAC_E6 * WAC_F**2

    # For testing the numerical inverse fitting
#    WAC_COEFF = np.zeros((4,4,2))
#    WAC_COEFF[0,1,0] = 1.5
#    WAC_COEFF[1,0,0] = 0.5
#    WAC_COEFF[0,1,1] = 0.75
#    WAC_COEFF[1,0,1] = 2.0
    
    DISTORTION_COEFF_XY_TO_UV = {'NAC': NAC_COEFF,
                                 'WAC': WAC_COEFF}
    
    # Create a master version of the inverse distortion model.
    # These coefficients were computed by numerically solving the above
    # polynomials.
    # These polynomials convert from U,V (pixels) to X,Y (radians).
    # Maximum errors from applying the original distortion model and
    # then inverting:
    # NAC
    #   X DIFF MIN MAX -7.01489382641e-10 1.15568299657e-10
    #   Y DIFF MIN MAX -7.7440587623e-10 8.81658628916e-10
    #   U DIFF MIN MAX -0.00138077186011 1.94695478513e-05
    #   V DIFF MIN MAX -0.000474712833352 0.000563339044731
    # WAC
    #   X DIFF MIN MAX -3.29122915029e-07 5.42305755497e-08
    #   Y DIFF MIN MAX -3.92109405705e-07 3.98905653939e-07
    #   U DIFF MIN MAX -0.0535697887647 0.000916852346108
    #   V DIFF MIN MAX -0.0182888189072 0.0187174796013
    
    NAC_INV_COEFF = np.zeros((4,4,2))
    NAC_INV_COEFF[:,:,0] = [[ -1.14799845e-10,  7.80494024e-14,  1.73312704e-15, -5.95242349e-19],
                            [  5.99190197e-06, -3.91823615e-13, -7.12054264e-15,  0.00000000e+00],
                            [  1.42455664e-12, -2.15242793e-18,  0.00000000e+00,  0.00000000e+00],
                            [ -7.11199158e-15,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]
    NAC_INV_COEFF[:,:,1] = [[ -4.86573092e-12,  5.99122009e-06, -3.91358768e-13, -7.12774967e-15],
                            [  1.03832114e-13,  1.41728538e-12, -1.55778300e-18,  0.00000000e+00],
                            [  2.03725690e-16, -7.11687723e-15,  0.00000000e+00,  0.00000000e+00],
                            [  1.67086926e-21,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]
    
    WAC_INV_COEFF = np.zeros((4,4,2))
    WAC_INV_COEFF[:,:,0] = [[ -5.42748411e-08,  5.06916433e-12,  8.16888725e-13, -3.85258837e-17],
                            [  5.97679707e-05, -3.53472340e-12, -5.13400740e-13,  0.00000000e+00],
                            [  5.66784001e-11, -1.27038656e-16,  0.00000000e+00,  0.00000000e+00],
                            [ -5.09498280e-13,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]
    WAC_INV_COEFF[:,:,1] = [[ -3.31749330e-10,  5.97618831e-05, -3.50726141e-12, -5.17011679e-13],
                            [  6.70738287e-12,  5.34696724e-11, -8.85892191e-17,  0.00000000e+00],
                            [  1.35389219e-14, -5.11900149e-13,  0.00000000e+00,  0.00000000e+00],
                            [  7.39668979e-19,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]

    # For testing the numerical inverse fitting
#    WAC_INV_COEFF[:,:,0] = [[  1.14377216e-09,  5.71428571e-01, -5.17435487e-16,  1.39385624e-17],
#                            [ -2.85714286e-01, -3.18326095e-14, -1.04402502e-17,  0.00000000e+00],
#                            [ -6.02467295e-14, -1.36440914e-16,  0.00000000e+00,  0.00000000e+00],
#                            [ -2.77233884e-17,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]
#    WAC_INV_COEFF[:,:,1] = [[ -2.86046684e-09, -1.90476190e-01,  1.72236899e-14, -3.66611741e-18],
#                            [  7.61904762e-01,  4.33089979e-14,  1.25388484e-16,  0.00000000e+00],
#                            [  7.96957986e-14,  1.53755115e-16,  0.00000000e+00,  0.00000000e+00],
#                            [  4.82114213e-17,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]
    
    DISTORTION_COEFF_UV_TO_XY = {'NAC': NAC_INV_COEFF,
                                 'WAC': WAC_INV_COEFF}
    
    ######################################################################
    
    @staticmethod
    def initialize():
        """Fills in key information about the WAC and NAC. Must be called
        first.
        """

        # Quick exit after first call
        if ISS.initialized: return

        Cassini.initialize()
        Cassini.load_instruments()

        # Load the instrument kernel
        ISS.instrument_kernel = Cassini.spice_instrument_kernel("ISS")[0]

        # Construct a Polynomial FOV for each camera
        for detector in ["NAC", "WAC"]:
            info = ISS.instrument_kernel["INS"]["CASSINI_ISS_" + detector]

            # Full field of view
            lines = info["PIXEL_LINES"]
            samples = info["PIXEL_SAMPLES"]

            xfov = info["FOV_REF_ANGLE"]
            yfov = info["FOV_CROSS_ANGLE"]
            assert info["FOV_ANGLE_UNITS"] == "DEGREES"
            
            uscale = np.arctan(np.tan(xfov * oops.RPD) / (samples/2.))
            vscale = np.arctan(np.tan(yfov * oops.RPD) / (lines/2.))
            
            # Display directions: [u,v] = [right,down]
            full_fov = oops.fov.Polynomial((samples,lines),
                                           coefft_uv_from_xy=
                                   ISS.DISTORTION_COEFF_XY_TO_UV[detector],
                                           coefft_xy_from_uv=None)
            full_fov_fast = oops.fov.Polynomial((samples,lines),
                                           coefft_uv_from_xy=
                                   ISS.DISTORTION_COEFF_XY_TO_UV[detector],
                                           coefft_xy_from_uv=
                                   ISS.DISTORTION_COEFF_UV_TO_XY[detector])
            full_fov_none = oops.fov.FlatFOV((uscale,vscale), (samples,lines))

            # Load the dictionary, include the subsampling modes
            ISS.fovs[detector, "FULL", False] = full_fov
            ISS.fovs[detector, "SUM2", False] = oops.fov.Subsampled(full_fov, 2)
            ISS.fovs[detector, "SUM4", False] = oops.fov.Subsampled(full_fov, 4)
            ISS.fovs[detector, "FULL", True] = full_fov_fast
            ISS.fovs[detector, "SUM2", True] = oops.fov.Subsampled(full_fov_fast, 2)
            ISS.fovs[detector, "SUM4", True] = oops.fov.Subsampled(full_fov_fast, 4)
            ISS.fovs[detector, "FULL", None] = full_fov_none
            ISS.fovs[detector, "SUM2", None] = oops.fov.Subsampled(full_fov_none, 2)
            ISS.fovs[detector, "SUM4", None] = oops.fov.Subsampled(full_fov_none, 4)

        # Construct a SpiceFrame for each camera
        # Deal with the fact that the instrument's internal coordinate system is
        # rotated 180 degrees
        nac_flipped = oops.frame.SpiceFrame("CASSINI_ISS_NAC",
                                            id="CASSINI_ISS_NAC_FLIPPED")
        wac_flipped = oops.frame.SpiceFrame("CASSINI_ISS_WAC",
                                            id="CASSINI_ISS_WAC_FLIPPED")

        rot180 = oops.Matrix3([[-1,0,0],[0,-1,0],[0,0,1]])
        ignore = oops.frame.Cmatrix(rot180, nac_flipped, id="CASSINI_ISS_NAC")
        ignore = oops.frame.Cmatrix(rot180, wac_flipped, id="CASSINI_ISS_WAC")

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

        snapshots = from_index(os.path.join(TESTDATA_PARENT_DIRECTORY, "cassini/ISS/index.lbl"))
        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY, "cassini/ISS/W1575634136_1.IMG"))
        snapshot3940 = snapshots[3940]  #should be same as snapshot
    
        self.assertTrue(abs(snapshot.time[0] - snapshot3940.time[0]) < 1.e-3)
        self.assertTrue(abs(snapshot.time[1] - snapshot3940.time[1]) < 1.e-3)

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
