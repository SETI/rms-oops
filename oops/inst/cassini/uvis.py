################################################################################
# oops/inst/cassini/uvis.py
################################################################################

import numpy as np
import pylab
import os
import sys
import csv
import julian
import pdstable
import pdsparser
import cspice
import oops

from oops.inst.cassini.cassini_ import Cassini

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, data=False):
    """A general, static method to return an Observation subclass object based
    on a label for a given Cassini UVIS file.

    Input:
        filespec        the full path to the PDS label of a UVIS data file.
        data            True to include the data array.
    """

    UVIS.initialize()   # Define everything the first time through

    # Load the PDS label
    lines = pdsparser.PdsLabel.load_file(filespec)

    # Deal with corrupt syntax
    for i in range(len(lines)):
        line = lines[i]
        if "CORE_UNIT" in line:
            if '"COUNTS/BIN"' not in line:
                lines[i] = re.sub( "COUNTS/BIN", '"COUNTS/BIN"', line)

    # Get dictionary
    label = pdsparser.PdsLabel.from_string(lines).as_dict()
    info = label["QUBE"]

    # Load any needed SPICE kernels
    tstart = julian.tdb_from_tai(julian.tai_from_iso(label["START_TIME"]))
    Cassini.load_cks( tstart, tstart + 3600.)
    Cassini.load_spks(tstart, tstart + 3600.)

    # Determine the detector
    det = label["PRODUCT_ID"]
    det = det[:det.index("2")]

    # Determine the slit state for EUV or FUV
    if det in ("EUV", "FUV"):
        slit_state = label["SLIT_STATE"]
    else:
        slit_state = ""

    # Define the FOV
    fov = UVIS.fovs[(det, slit_state)]
    line0 = info["UL_CORNER_LINE"]
    line1 = info["LR_CORNER_LINE"]
    if (line0,line1) != (0,63):
        fov = oops.fov.Subarray(fov, (line0,0), (line1-line0,0))

    binning = info["LINE_BIN"]
    if binning != 1:
        fov = oops.fov.SliceFOV(fov, (binning,1))

    # Define the cadence
    texp = label["INTEGRATION_DURATION"]
    samples = info["CORE_ITEMS"][2]

    cadence = oops.cadence.Metronome(tstart, texp, texp, samples)

    # Define the coordinate frame
    frame_id = UVIS.frames[(det, slit_state)]

    # Load the data array
    (bands,lines,samples) = info["CORE_ITEMS"]

    if data:
        head = os.path.split(data_filespec)[0]
        body = label["^QUBE"]

        data_filespec = os.path.join(head, body)
        if not os.path.exists(data_filespec):
            data_filespec = os.path.join(head, body.lower())

        # This can be generalized if necessary...
        assert info["CORE_ITEM_TYPE"] == "MSB_UNSIGNED_INTEGER"
        assert info["CORE_ITEM_BYTES"] == 2
        assert info["SUFFIX_ITEMS"] == (0,0,0)

        array = np.from_file(data_filespec, sep="", dtype=">i2")

        if lines == 1:
            new_shape = (samples,bands)
        else:
            new_shape = (samples,lines,bands)

        array = array.reshape(new_shape)

    # Create the Observation
    if lines == 1:
        obs = oops.obs.Pixel(("t","b"), cadence, fov, "CASSINI", frame_id)
    else:
        obs = oops.obs.Slit(("vt","u","b"), 1, cadence, fov, "CASSINI",
                                                             frame_id)

    if data:
        obs.insert_subfield("data", array)

    return obs

################################################################################

class UVIS(object):
    """A instance-free class to hold Cassini UVIS instrument parameters."""

    instrument_kernel = None
    fovs = {}
    initialized = False

    abbrevs = {"FUV_HI":  ("FUV", "HIGH_RESOLUTION"),
               "FUV_LO":  ("FUV", "LOW_RESOLUTION"),
               "FUV_OCC": ("FUV", "OCCULTATION"),
               "EUV_HI":  ("EUV", "HIGH_RESOLUTION"),
               "EUV_LO":  ("EUV", "LOW_RESOLUTION"),
               "EUV_OCC": ("EUV", "OCCULTATION"),
               "HSP":     ("HSP", ""),
               "HDAC":    ("HDAC", "")}

    frame_ids = {("FUV", "HIGH_RESOLUTION"): "CASSINI_UVIS_FUV",
                 ("FUV", "LOW_RESOLUTION") : "CASSINI_UVIS_FUV",
                 ("FUV", "OCCULTATION")    : "CASSINI_UVIS_FUV",
                 ("EUV", "HIGH_RESOLUTION"): "CASSINI_UVIS_EUV",
                 ("EUV", "LOW_RESOLUTION") : "CASSINI_UVIS_EUV",
                 ("EUV", "OCCULTATION")    : "CASSINI_UVIS_EUV",
                 ("HSP", "")               : "CASSINI_UVIS_HSP",
                 ("HDAC", "")              : "CASSINI_UVIS_HDAC"}

    @staticmethod
    def initialize():
        """Fills in key information about the UVIS channels. Must be called
        first.
        """

        # Quick exit after first call
        if UVIS.initialized: return

        Cassini.initialize()
        Cassini.load_instruments()

        # Load the instrument kernel
        UVIS.instrument_kernel = Cassini.spice_instrument_kernel("UVIS")[0]

        # Construct a flat FOV for each detector
        for detector in ("FUV_HI", "FUV_LO", "FUV_OCC",
                         "EUV_HI", "EUV_LO", "EUV_OCC",
                         "SOLAR", "HSP", "HDAC"):
            info = UVIS.instrument_kernel["INS"]["CASSINI_UVIS_" + detector]

            # Full field of view
            lines = info["PIXEL_LINES"]
            samples = info["PIXEL_SAMPLES"]

            xfov = info["FOV_REF_ANGLE"]
            yfov = info["FOV_CROSS_ANGLE"]
            assert info["FOV_ANGLE_UNITS"] == "DEGREES"

            uscale = np.arctan(np.tan(xfov * oops.RPD) / 32.)
            vscale = np.arctan(np.tan(yfov * oops.RPD) / 32.)

            # Display directions: [u,v] = [right,down]
            normal_fov = oops.fov.Flat((uscale,vscale), (64,64))
            hires_fov  = oops.fov.Flat((uscale / det[2], vscale / det[3]),
                                       (64 * det[2], 64 * det[3]))

            # Load the dictionary
            VIMS.fovs[(det[1],"NORMAL")] = normal_fov
            VIMS.fovs[(det[1],"HI-RES")] = hires_fov
            VIMS.fovs[(det[1],"N/A")] = normal_fov      # just prevents KeyError

        # Construct a SpiceFrame for each detector
        ignore = oops.frame.SpiceFrame("CASSINI_UVIS_FUV")
        ignore = oops.frame.SpiceFrame("CASSINI_UVIS_EUV")
        ignore = oops.frame.SpiceFrame("CASSINI_UVIS_SOLAR")
        ignore = oops.frame.SpiceFrame("CASSINI_UVIS_SOL_OFF")
        ignore = oops.frame.SpiceFrame("CASSINI_UVIS_HSP")
        ignore = oops.frame.SpiceFrame("CASSINI_UVIS_HDAC")

        UVIS.initialized = True

    @staticmethod
    def reset():
        """Resets the internal Cassini UVIS parameters. Can be useful for
        debugging."""

        UVIS.instrument_kernel = None
        UVIS.fovs = {}
        UVIS.initialized = False

        Cassini.reset()

################################################################################
# Initialize at load time
################################################################################

UVIS.initialize()

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Cassini_UVIS(unittest.TestCase):
    
    def runTest(self):

        # TBD
        pass

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################


