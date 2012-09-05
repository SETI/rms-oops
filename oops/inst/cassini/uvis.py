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
from oops_.array.all import *
from oops_.meshgrid import Meshgrid

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

    # Load the data array
    (bands,lines,samples) = info["CORE_ITEMS"]

    # Determine the detector
    det = label["PRODUCT_ID"]
    det = det[:det.index("2")]

    # Determine the slit state for EUV or FUV
    if det in ("EUV", "FUV"):
        slit_state = label["SLIT_STATE"]
    else:
        slit_state = ""

    # Define the FOV
    uv_shape = (lines, 1)
    uscale = np.arctan(np.tan(UVIS.xfovs[(det, slit_state)] * oops.RPD) / 32.)
    vscale = np.arctan(np.tan(UVIS.yfovs[(det, slit_state)] * oops.RPD) / 32.)
    fov = oops.fov.Flat((uscale,vscale), uv_shape)
    print "uv_shape:", uv_shape

    #print "det = ", det
    #print "slit_state =", slit_state
    #fov = UVIS.fovs[(det, slit_state)]
    print "fov after getting from cerating originally:", fov
    line0 = info["UL_CORNER_LINE"]
    line1 = info["LR_CORNER_LINE"]
    print "line0, line1:", line0, line1
    if (line0,line1) != (0,63):
        #fov = oops.fov.Subarray(fov, (line0,0), (line1-line0,0))
        #shape should have 1 in v direction, not 0 (which would lack dimension)
        fov = oops.fov.Subarray(fov, (line0,0), (line1-line0,1))
        print "fov after creating subarray:", fov

    binning = info["LINE_BIN"]
    if binning != 1:
        fov = oops.fov.SliceFOV(fov, (binning,1))
        print "fov after getting SliceFOV:", fov

    # Define the cadence
    texp = label["INTEGRATION_DURATION"]
    samples = info["CORE_ITEMS"][2]

    cadence = oops.cadence.Metronome(tstart, texp, texp, samples)
    print "cadence.steps:", cadence.steps

    # Define the coordinate frame
    frame_id = UVIS.frame_ids[(det, slit_state)]

    if data:
        head = os.path.split(filespec)[0]
        body = label["^QUBE"]

        data_filespec = os.path.join(head, body)
        if not os.path.exists(data_filespec):
            data_filespec = os.path.join(head, body.lower())

        # This can be generalized if necessary...
        assert info["CORE_ITEM_TYPE"] == "MSB_UNSIGNED_INTEGER"
        assert info["CORE_ITEM_BYTES"] == 2
        assert info["SUFFIX_ITEMS"] == [0,0,0]

        array = np.fromfile(data_filespec, sep="", dtype=">i2")

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
    obs.insert_subfield("dict", label)
    obs.insert_subfield("index_dict", label)# for backward compatibility
    obs.insert_subfield("instrument", "UVIS")
    obs.insert_subfield("detector", det)
    obs.insert_subfield("sampling", slit_state)

    if data:
        obs.insert_subfield("data", array)

    return obs

def meshgrid_and_times(obs):
    """Returns a meshgrid object and time array that oversamples and extends the
        dimensions of the field of view of a UVIS observation.
        
        Input:
        obs             the UVIS observation object to for which to generate a
                        meshgrid and a time array.
        
        Return:         (mesgrid, time)
        """
    
    swap = obs.u_axis > obs.v_axis or obs.u_axis == -1
    meshgrid = Meshgrid.for_fov(obs.fov, swap=swap)
    uv_shape = meshgrid.uv.shape
    print "\n************\nuv_shape =", uv_shape
    print "meshgrid.uv.vals.shape =", meshgrid.uv.vals.shape
    meshgrid.uv = Pair.as_pair(meshgrid.uv.vals.reshape(uv_shape[0],1,2))
    #meshgrid.uv.reshape((meshgrid.uv.shape,1))
    print "in UVIS:meshgrid_and_times: meshgrid.uv.shape =", meshgrid.uv.shape

    time = Scalar.as_scalar(np.linspace(obs.cadence.tstart,
                                    obs.cadence.tstart +
                                    obs.cadence.steps * obs.cadence.tstride,
                                    obs.cadence.steps).reshape(1,obs.cadence.steps))
    
    return (meshgrid, time)

################################################################################

class UVIS(object):
    """A instance-free class to hold Cassini UVIS instrument parameters."""

    instrument_kernel = None
    fovs = {}
    initialized = False
    xfovs = {}
    yfovs = {}

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
        
        """info = UVIS.instrument_kernel["INS"]["CASSINI_UVIS_FUV"]
        print "FUV"
        for key in info:
            print "info key, value:", key, info[key]
        info = UVIS.instrument_kernel["INS"]["CASSINI_UVIS_EUV"]
        print "EUV"
        for key in info:
            print "info key, value:", key, info[key]
        info = UVIS.instrument_kernel["INS"]["CASSINI_UVIS_HSP"]
        print "HSP"
        for key in info:
            print "info key, value:", key, info[key]
        info = UVIS.instrument_kernel["INS"]["CASSINI_UVIS_HDAC"]
        print "HDAC"
        for key in info:
            print "info key, value:", key, info[key]"""

        # Construct a flat FOV for each detector
        # the detectors returned from the call to Cassini.spice_instrument_kernel
        #for detector in ("FUV_HI", "FUV_LO", "FUV_OCC",
        #                 "EUV_HI", "EUV_LO", "EUV_OCC",
        #                 "SOLAR", "HSP", "HDAC"):
        for detector in ("FUV_HI", "FUV_LO", "FUV_OCC",
                         "EUV_HI", "EUV_LO", "EUV_OCC",
                         "HSP"):
            #info = UVIS.instrument_kernel["INS"]["CASSINI_UVIS_" + detector]
            # "INS" values do not contain the "HI" or "LO" or "OCC" part, but
            # "BODY" values do.
            #info = UVIS.instrument_kernel["BODY"]["CASSINI_UVIS_" + detector]
            # however, we want the data available from "INS", and we'll have to
            # calculate the _LO and _OCC values
            instrument = UVIS.abbrevs[detector][0]
            resolution = UVIS.abbrevs[detector][1]
            info = UVIS.instrument_kernel["INS"]["CASSINI_UVIS_" + instrument]
            UVIS.xfovs[(instrument,resolution)] = info["FOV_CROSS_ANGLE"]
            UVIS.yfovs[(instrument,resolution)] = info["FOV_REF_ANGLE"]

            # check what we have
            #for key in info:
            #    print "info key, value:", key, info[key]
            # Full field of view
            """lines = info["PIXEL_LINES"]
            samples = info["PIXEL_SAMPLES"]
            

            yfov = info["FOV_REF_ANGLE"]
            xfov = info["FOV_CROSS_ANGLE"]
            xpixels = 64
            if resolution == "LOW_RESOLUTION":
                xfov *= 2.
                xpixels = 32
            elif resolution == "OCCULTATION":
                xfov = 0.24268
                xpixels = 6
            assert info["FOV_ANGLE_UNITS"] == "DEGREES"

            uscale = np.arctan(np.tan(xfov * oops.RPD) / 32.)
            vscale = np.arctan(np.tan(yfov * oops.RPD) / 32.)

            # Display directions: [u,v] = [right,down]
            # numbers are for hires
            UVIS.fovs[(instrument,resolution)] = oops.fov.Flat((uscale,vscale), (xpixels,1))
            #UVIS.fovs[(det[1],"N/A")] = normal_fov      # just prevents KeyError
            """

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
        UVIS.xfovs = {}
        UVIS.yfovs = {}
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


