################################################################################
# oops/inst/cassini/utils.py
#
# Utility functions for managing SPICE kernels while working with Cassini data
# sets.
################################################################################

import numpy as np
import unittest
import os.path

import julian
import textkernel
import spicedb
import cspice
import oops

################################################################################
# Routines for managing the loading of C and SP kernels
################################################################################

# Make sure the leap seconds have been loaded
oops.spice.load_leap_seconds()

# We load CK and SPK files on a very rough month-by-month basis. This is simpler
# than a more granular approach involving detailed calendar calculations. We
# divide the period October 1, 1997 to January 1, 2018 up into 243 "months" of
# equal length. Given any TDB, we quickly determine the month within which it
# falls. Each month is associated with a list of kernels that should be loaded
# whenever information is needed about any time within that month +/- 12 hours.
# The kernels needed for a given month only get loaded when they are needed, and
# are only loaded once. For any geometry calculation involving Cassini, a quick
# call to load_ck(time) or load_spk(time) will ensure that the information is
# available.

################################################################################

class Cassini(object):
    """A instance-free class to hold Cassini-specific parameters."""

    START_TIME = "1997-10-01"
    STOP_TIME  = "2018-01-01"
    
    TDB0 = julian.tdb_from_tai(julian.tai_from_iso(START_TIME))
    TDB1 = julian.tdb_from_tai(julian.tai_from_iso(STOP_TIME))
    MONTHS = 243
    DTDB = (TDB1 - TDB0) / MONTHS
    SLOP = 43200.
    
    CK_SUBPATH = os.path.join("Cassini", "CK-reconstructed")
    CK_LOADED  = np.zeros(MONTHS, dtype="bool")  # True if month was loaded
    CK_LIST    = np.empty(MONTHS, dtype="object")# Kernels needed
    CK_DICT    = {}     # A dictionary of kernel names returning True if loaded
    
    SPK_SUBPATH = os.path.join("Cassini", "SPK-reconstructed")
    SPK_PREDICT = os.path.join("Cassini", "SPK-predicted")
    SPK_LOADED  = np.zeros(MONTHS, dtype="bool")
    SPK_LIST    = np.empty(MONTHS, dtype="object")
    SPK_DICT    = {}

    loaded_instruments = []

    initialized = False

    ############################################################################

    @staticmethod
    def initialize():
        if Cassini.initialized: return

        spicedb.open_db()

        kernels = spicedb.select_kernels("CK", path=Cassini.CK_SUBPATH,
                                         time=("1000-01-01", "2999-12-31"))
        Cassini.initialize_kernels(kernels, Cassini.CK_LIST, Cassini.CK_DICT)

        kernels = spicedb.select_kernels("SPK", path=Cassini.SPK_SUBPATH,
                                          time=("1000-01-01", "2999-12-31"))
        Cassini.initialize_kernels(kernels, Cassini.SPK_LIST, Cassini.SPK_DICT)

        spicedb.close_db()

        # Define some important paths and frames
        oops.define_solar_system(Cassini.START_TIME, Cassini.STOP_TIME)
        ignore = oops.path.SpicePath("CASSINI", "SATURN")
        #ignore = oops.frame.SpiceFrame("CASSINI_SC_COORD", "J2000")

        initialized = True

    @staticmethod
    def reset():
        """Resets the internal parameters. Can be useful for debugging."""

        Cassini.loaded_instruments = []

        Cassini.CK_LOADED = np.zeros(Cassini.MONTHS, dtype="bool")
        Cassini.CK_LIST = np.empty(Cassini.MONTHS, dtype="object")
        Cassini.CK_DICT = {}
        
        Cassini.SPK_LOADED = np.zeros(Cassini.MONTHS, dtype="bool")
        Cassini.SPK_LIST = np.empty(Cassini.MONTHS, dtype="object")
        Cassini.SPK_DICT = {}

        Cassini.initialized = False

    ############################################################################

    @staticmethod
    def load_ck(t):
        """Ensures that the C kernels applicable at or near the given time have
        been furnished. The time can be tai or tdb."""

        Cassini.load_kernels(t, t, Cassini.CK_LOADED, Cassini.CK_LIST,
                                   Cassini.CK_DICT)

    @staticmethod
    def load_cks(t0, t1):
        """Ensures that all the C kernels applicable near or within the time
        interval tdb0 to tdb1 have been furnished. The time can be tai or tdb.
        """

        Cassini.load_kernels(t0, t1, Cassini.CK_LOADED, Cassini.CK_LIST,
                                     Cassini.CK_DICT)

    @staticmethod
    def load_spk(t):
        """Ensures that the SPK kernels applicable at or near the given time have
        been furnished. The time can be tai or tdb."""

        Cassini.load_kernels(t, t, Cassini.SPK_LOADED, Cassini.SPK_LIST,
                                   Cassini.SPK_DICT)

    @staticmethod
    def load_spks(t0, t1):
        """Ensures that all the SPK kernels applicable near or within the time
        interval tdb0 to tdb1 have been furnished. The time can be tai or tdb."""

        Cassini.load_kernels(t0, t1, Cassini.SPK_LOADED, Cassini.SPK_LIST,
                                     Cassini.SPK_DICT)

    @staticmethod
    def load_kernels(t0, t1, loaded, lists, dict):

        spice_path = spicedb.get_spice_path()

        # Find the range of months needed
        m1 = int((t0 - Cassini.TDB0) // Cassini.DTDB)
        m2 = int((t1 - Cassini.TDB0) // Cassini.DTDB) + 1

        # Load any months not already loaded
        for m in range(m1, m2):
            if not loaded[m]:
                for name in lists[m]:
                    if not dict[name]:
                        cspice.furnsh(os.path.join(spice_path, name))
                        dict[name] = True
                loaded[m] = True

    ########################################
    # Initialize the kernel lists
    ########################################

    @staticmethod
    def initialize_kernels(kernels, lists, dict):

        for i in range(Cassini.MONTHS):
            lists[i] = []

        for kernel in kernels:

            # Add the kernel's filespec to the dictionary
            dict[kernel.filespec] = False

            # Find the range of months applicable, extended by 12 hours
            t0 = cspice.str2et(kernel.start_time) - Cassini.SLOP
            t1  = cspice.str2et(kernel.stop_time) + Cassini.SLOP

            m1 = int((t0 - Cassini.TDB0) // Cassini.DTDB)
            m2 = int((t1 - Cassini.TDB0) // Cassini.DTDB) + 1

            # Add this kernel to each month's list
            for m in range(m1, m2):
                lists[m] += [kernel.filespec]

    ############################################################################
    # Routines for managing the loading other kernels
    ############################################################################

    @staticmethod
    def load_instruments(instruments=[], asof=None):
        """Loads the SPICE kernels and defines the basic paths and frames for
        the Cassini mission. It is generally only be called once.

        Input:
            instruments an optional list of instrument names for which to load
                        frames kernels. The frames for ISS, VIMS, UVIS, and CIRS
                        are always loaded.

            asof        if this specifies a date or date-time in ISO format,
                        then only kernels that existed before the specified date
                        are used. Otherwise, the most recent versions are always
                        loaded.
        """

        # Load the default instruments on the first pass
        if Cassini.loaded_instruments == []:
            instruments += ["ISS", "VIMS", "CIRS", "UVIS"]

        # On later calls, return quickly if there's nothing to do
        if instruments == []: return

        # Check the formatting of the "as of" date
        if asof is not None:
            (day, sec) = julian.day_sec_from_iso(AS_OF)
            asof = julian.ymdhms_format_from_day_sec(day, sec)

        spicedb.open_db()

        # Select instrument frames
        kernels = []
        for inst in instruments:
            kernels += spicedb.select_kernels("IK", asof=asof,
                                                    name="CAS_" + inst + "%")

        # On the first pass, add the mission-wide kernels
        if Cassini.loaded_instruments == []:

            # Select spacecraft frames
            kernels += spicedb.select_kernels("FK", asof=asof,
                                                    name="CAS_V%")

            kernels += spicedb.select_kernels("FK", asof=asof,
                                                    name="CAS_STATUS_V%")

            # Select the spacecraft clock kernel
            kernels += spicedb.select_kernels("SCLK", asof=asof, body=-82)

            # Include the predict SPK
            kernels += spicedb.select_kernels("SPK", asof=asof, body=-82,
                                                     path=Cassini.SPK_PREDICT)

            # Also make sure leap seconds have been loaded
            oops.spice.load_leap_seconds()

        # Furnish everything
        ignore = spicedb.furnish_kernels(kernels)
        Cassini.loaded_instruments += instruments
 
        spicedb.close_db()

    ############################################################################
    # Routines for managing text kernel information
    ############################################################################

    @staticmethod
    def spice_instrument_kernel(inst_name, asof=None, name=None):
        """Returns a dictionary containing the Instrument Kernel information for
        a particular Cassini instrument. It also furnishes it for use by the
        SPICE tools.

        Input:
            inst_name   one of "ISS", "UVIS", "VIMS", "CIRS", etc.
            asof        an optional date in the past, in ISO date or date-time
                        format. If provided, then the information provided will
                        be applicable as of that date. Otherwise, the most
                        recent information is always provided.
            name        if provided, this is the name of the kernel to be
                        returned. Otherwise, the most recent version is
                        returned.

        Return:         a tuple containing:
                            the dictionary generated by textkernel.from_file()
                            the name of the kernel.
        """

        if name is None:
            name = "CAS_" + inst_name.upper() + "%"

        if asof is not None:
            (day,sec) = julian.day_sec_from_iso(stop_time)
            asof = julian.ymdhms_format_from_day_sec(day, sec)

        spicedb.open_db()
        kernel_info = spicedb.select_kernels("IK", name=name, asof=asof)
        spicedb.furnish_kernels(kernel_info)
        spicedb.close_db()

        return (spicedb.as_dict(kernel_info), spicedb.as_names(kernel_info)[0])

    ############################################################################

    @staticmethod
    def spice_frames_kernel(asof=None, name=None):
        """Returns a dictionary containing the Cassini Frames Kernel
        information. It also furnishes the kernels for use by the SPICE tools.

        Input:
            asof        an optional date in the past, in ISO date or date-time
                        format. If provided, then the information provided will
                        be applicable as of that date. Otherwise, the most
                        recent information is always provided.
            name        if provided, this is the name of the kernel to be
                        returned. Otherwise, the most recent version is
                        returned.

        Return:         a tuple containing:
                            the dictionary generated by textkernel.from_file()
                            an ordered list of the names of the kernels
        """

        if asof is not None:
            (day,sec) = julian.day_sec_from_iso(stop_time)
            asof = julian.ymdhms_format_from_day_sec(day, sec)

        spicedb.open_db()
        kernel_list = []
        kernel_list += select_kernels("FK", asof=asof, name="CAS_V%")
        kernel_list += select_kernels("FK", asof=asof, name="CAS_STATUS_V%")

        if name is not None:
            kernel_list += select_kernels("FK", name=name)

        spicedb.close_db()

        return (spicedb.as_dict(kernel_list), spicedb.as_names(kernel_list))

################################################################################
