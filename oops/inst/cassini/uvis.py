################################################################################
# oops/inst/cassini/uvis.py
################################################################################

import oops
import numpy as np
import os
import julian
import pdsparser
import cspice

from oops.inst.cassini.cassini_ import Cassini

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, data=False):
    """A general, static method to return one or more Observation subclass
    objects based on a label for a given Cassini UVIS file.

    Input:
        filespec        the full path to the PDS label of a UVIS data file.
        data            True to include the data array.
    """

    DEBUG = True        # True to assert that the data array must have null
                        # values outside the active windows

    UVIS.initialize()   # Define everything the first time through

    # Load the PDS label
    recs = pdsparser.PdsLabel.load_file(filespec)

    # Deal with corrupt syntax
    for i in range(len(recs)):
        rec = recs[i]
        if "CORE_UNIT" in rec:
            if '"COUNTS/BIN"' not in rec:
                recs[i] = re.sub( "COUNTS/BIN", '"COUNTS/BIN"', rec)

    # Get the label dictionary and data array dimensions
    label = pdsparser.PdsLabel.from_string(recs).as_dict()

    # Load any needed SPICE kernels
    tstart = julian.tdb_from_tai(julian.tai_from_iso(label["START_TIME"]))
    tstop  = julian.tdb_from_tai(julian.tai_from_iso(label["STOP_TIME"]))
    Cassini.load_cks( tstart, tstop)
    Cassini.load_spks(tstart, tstop)

    # Determine the detector used
    detector = label["PRODUCT_ID"]
    detector = detector[:detector.index("2")]   # The year always begins with 2

    # Define the instrument frame
    frame_id = UVIS.frame_ids[detector]

    # Define the full FOV
    if detector in ("EUV", "FUV"):
        resolution = label["SLIT_STATE"]
    else:
        resolution = ""

    # Get array dimensions
    if "QUBE" in label.keys():
        object = "QUBE"
        info = label[object]
        (bands,lines,samples) = info["CORE_ITEMS"]
        assert lines in (1,64)
        if lines > 1:
            shape = (lines, samples, bands)
        else:
            shape = (samples, bands)
        texp = label["INTEGRATION_DURATION"]
    else:
        object = "TIME_SERIES"
        info = label[object]
        samples = info["ROWS"]
        lines = 1
        bands = 1
        shape = (samples,)
        assert info["COLUMNS"] == 1
        assert (info["SAMPLING_PARAMETER_UNIT"] == "MILLISECOND" or
                info["SAMPLING_PARAMETER_UNIT"] == "MILLISECONDS")
        texp = info["SAMPLING_PARAMETER_INTERVAL"] * 0.001

    info = label[object]

    # Define the FOV and cadence
    fov = UVIS.fovs[(detector, resolution, lines)]
    cadence = oops.cadence.Metronome(tstart, texp, texp, samples)

    # Load the data array if necessary
    if data:
        head = os.path.split(filespec)[0]
        body = label["^" + object]

        data_filespec = os.path.join(head, body)
        if not os.path.exists(data_filespec):
            test_filespec = os.path.join(head, body.lower())
            if not os.path.exists(test_filespec):
                f = open(data_filespec,"r") # raise IOError

        # This can be generalized, but it works for the archived volumes...
        if object == "QUBE":
            assert info["CORE_ITEM_TYPE"] == "MSB_UNSIGNED_INTEGER"
            assert info["CORE_ITEM_BYTES"] == 2
            assert info["SUFFIX_ITEMS"] == [0,0,0]
            array_null = 65535              # Incorrectly -1 in many labels
        else:
            column = info["PHOTOMETER_COUNTS"]
            assert column["DATA_TYPE"] == "MSB_UNSIGNED_INTEGER"
            assert column["BYTES"] == 2
            array_null = None

        array = np.fromfile(data_filespec, sep="", dtype=">u2")

        # Re-shape into something sensible
        # Note that the axis order in the label is first-index-fastest
        if lines > 1:
            array = array.reshape((samples,lines,bands))
            array = array.swapaxes(0,1)     # now (lines, samples, bands)
        elif bands > 1:
            array = array.reshape((samples,bands))
        else:
            pass                            # no reshape needed

    # Identify the window(s) used
    # Note that these are either integers or tuples of integers
    if object == "QUBE":
        lwindow0 = info["UL_CORNER_LINE"]
        lwindow1 = info["LR_CORNER_LINE"]
        lbinning = info["LINE_BIN"]

        bwindow0 = info["UL_CORNER_BAND"]
        bwindow1 = info["LR_CORNER_BAND"]
        bbinning = info["BAND_BIN"]

        if type(lwindow0) == type(()):
            windows = len(lwindow0)
            assert len(lwindow1) == windows
            assert len(lbinning) == windows
            assert len(bwindow0) == windows
            assert len(bwindow1) == windows
            assert len(wbinning) == windows
        else:
            windows = 1
            lwindow0 = (lwindow0,)
            lwindow1 = (lwindow1,)
            lbinning = (lbinning,)
            bwindow0 = (bwindow0,)
            bwindow1 = (bwindow1,)
            bbinning = (bbinning,)

        # Check the outer periphery of the data array in DEBUG mode
        if DEBUG and data:
            l0 = lwindow0[0]
            l1 = lwindow1[-1] + 1
            assert np.all(array[:l0,  ...] == array_null)
            assert np.all(array[ l1:, ...] == array_null)

            b0 = bwindow0[0]
            b1 = bwindow1[-1] + 1
            assert np.all(array[..., :b0 ] == array_null)
            assert np.all(array[...,  b1:] == array_null)
    else:
        windows = 1         # no windows or resampling in HSP or HDAC
        lwindow0 = (0,)
        lwindow1 = (lines-1,)
        lbinning = (1,)
        bwindow0 = (0,)
        bwindow1 = (bands-1,)
        bbinning = (1,)

    # For each window...
    obslist = []
    for w in range(windows):

        obs_fov = fov

        if data:
            slice = array

        # Trim the lines
        l0 = lwindow0[w]
        l1 = lwindow1[w] + 1
        dl = l1 - l0

        if (l0,l1) != (0,lines):
            obs_fov = oops.fov.SliceFOV(obs_fov, (0,l0), (1,dl))
            shape = (dl,) + shape[1:]

            if data:
                slice = slice[l0:l1, :]

        # Trim the bands
        b0 = bwindow0[w]
        b1 = bwindow1[w] + 1
        db = b1 - b0

        if (b0,b1) != (0,bands):
            shape = shape[:-1] + (db,)

            if data:
                slice = slice[..., b0:b1]

        # Bin the lines
        lbin = lbinning[w]
        if lbin != 1:
            assert dl % lbin == 0
            obs_fov = oops.fov.Subsampled(obs_fov, (1,lbin))
            dl_scaled = dl / lbin
            shape = (dl_scaled,) + shape[1:]

            if data:
                if DEBUG:
                    assert np.all(slice[dl_scaled:, ...] == array_null)

                slice = slice[:dl_scaled]

        # Bin the bands
        bbin = bbinning[w]
        if bbin != 1:
            assert db % bbin == 0
            db_scaled = db / bbin
            shape = shape[:-1] + (db_scaled,)

            if data:
                if DEBUG:
                    assert np.all(slice[..., db_scaled:] == array_null)

                slice = slice[..., :db_scaled]

        # Create the Observation
        if lines == 1:
            obs = oops.obs.Pixel(("t"), cadence, obs_fov, "CASSINI", frame_id)
        else:
            obs = oops.obs.Slit(("v","ut","b"), 1, cadence, obs_fov,
                                 "CASSINI", frame_id)

        obs.shape = shape       # Override the band dimension

        obs.insert_subfield("dict", label)
        obs.insert_subfield("instrument", "UVIS")
        obs.insert_subfield("detector", detector)
        obs.insert_subfield("sampling", resolution)
        obs.insert_subfield("line_window", (l0,l1))
        obs.insert_subfield("band_window", (b0,b1))
        obs.insert_subfield("line_binning", lbin)
        obs.insert_subfield("band_binning", bbin)

        if data:
            obs.insert_subfield("data", slice)

        obslist.append(obs)

    return tuple(obslist)

################################################################################

class UVIS(object):
    """A instance-free class to hold Cassini UVIS instrument parameters."""

    instrument_kernel = None
    fovs = {}
    initialized = False

    # Map NAIF body names (following "CASSINI_UVIS_") to (detector, resolution)
    abbrevs = {"CASSINI_UVIS_FUV_HI" : ("FUV", "HIGH_RESOLUTION"),
               "CASSINI_UVIS_FUV_LO" : ("FUV", "LOW_RESOLUTION"),
               "CASSINI_UVIS_FUV_OCC": ("FUV", "OCCULTATION"),
               "CASSINI_UVIS_EUV_HI" : ("EUV", "HIGH_RESOLUTION"),
               "CASSINI_UVIS_EUV_LO" : ("EUV", "LOW_RESOLUTION"),
               "CASSINI_UVIS_EUV_OCC": ("EUV", "OCCULTATION"),
               "CASSINI_UVIS_SOLAR"  : ("SOLAR",   ""),
               "CASSINI_UVIS_SOL_OFF": ("SOL_OFF", ""),
               "CASSINI_UVIS_HSP"    : ("HSP",     ""),
               "CASSINI_UVIS_HDAC"   : ("HDAC",    "")}

    # Map detector to NAIF frame ID
    frame_ids = {"FUV"    : "CASSINI_UVIS_FUV",
                 "EUV"    : "CASSINI_UVIS_EUV",
                 "SOLAR"  : "CASSINI_UVIS_SOLAR",
                 "SOL_OFF": "CASSINI_UVIS_SOL_OFF",
                 "HSP"    : "CASSINI_UVIS_HSP",
                 "HDAC"   : "CASSINI_UVIS_HDAC"}

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

        # TEMPORARY FIX to a loader problem
        ins = UVIS.instrument_kernel["INS"]
        ins['CASSINI_UVIS_FUV_HI']  = ins[-82840]
        ins['CASSINI_UVIS_FUV_LO']  = ins[-82841]
        ins['CASSINI_UVIS_FUV_OCC'] = ins[-82842]
        ins['CASSINI_UVIS_EUV_HI']  = ins[-82843]
        ins['CASSINI_UVIS_EUV_LO']  = ins[-82844]
        ins['CASSINI_UVIS_EUV_OCC'] = ins[-82845]
        ins['CASSINI_UVIS_HSP']     = ins[-82846]
        ins['CASSINI_UVIS_HDAC']    = ins[-82847]
        ins['CASSINI_UVIS_SOLAR']   = ins[-82848]
        ins['CASSINI_UVIS_SOL_OFF'] = ins[-82848]

        # Construct a flat FOV and load the frame for each detector
        for key in UVIS.abbrevs.keys():
            (detector, resolution) = UVIS.abbrevs[key]

            # Construct the SpiceFrame
            ignore = oops.frame.SpiceFrame(UVIS.frame_ids[detector])

            # Get the FOV angles
            info = UVIS.instrument_kernel["INS"][key]

            if info["FOV_SHAPE"] == "RECTANGLE":
                u_angle = 2. * info["FOV_CROSS_ANGLE"] * oops.RPD
                v_angle = 2. * info["FOV_REF_ANGLE"] * oops.RPD
            elif info["FOV_SHAPE"] == "CIRCLE":
                u_angle = 2. * info["FOV_REF_ANGLE"] * oops.RPD
                v_angle = u_angle
            else:
                raise RuntimeError("Unrecognized FOV_SHAPE: " +
                                   info["FOV_SHAPE"])

            # Define the frame for 1 or 64 lines
            # Not every combination is really used but that doesn't matter
            for lines in {1, 64}:
                fov = oops.fov.Flat((u_angle, v_angle/lines), (1,lines))
                UVIS.fovs[(detector, resolution, lines)] = fov

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


