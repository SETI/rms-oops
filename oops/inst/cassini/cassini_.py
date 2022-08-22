################################################################################
# oops/inst/cassini/cassini_.py
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
import cspyce
import oops

import oops.body as body

TOUR = (2003 - 2000) * 365 * 86400      # Rough ET dividing Saturn from Jupiter

################################################################################
# Routines for managing the loading of C and SP kernels
################################################################################

# Make sure the leap seconds have been loaded
oops.spice.load_leap_seconds()

# We load CK and SPK files on a very rough month-by-month basis. This is simpler
# than a more granular approach involving detailed calendar calculations. We
# divide the period October 1, 1997 to October 1, 2017 up into 240 "months" of
# equal length. Given any TDB, we quickly determine the month within which it
# falls. Each month is associated with a list of kernels that should be loaded
# whenever information is needed about any time within that month +/- 12 hours.
# The kernels needed for a given month only get loaded when they are needed, and
# are only loaded once. For any geometry calculation involving Cassini, a quick
# call to load_ck(time) or load_spk(time) will ensure that the information is
# available.

################################################################################

#*******************************************************************************
# Cassini class
#*******************************************************************************
class Cassini(object):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A instance-free class to hold Cassini-specific parameters.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    START_TIME = "1997-10-01"
    STOP_TIME  = "2017-10-01"
    MONTHS = 240        # 20 years * 12 months/year

    TDB0 = julian.tdb_from_tai(julian.tai_from_iso(START_TIME))
    TDB1 = julian.tdb_from_tai(julian.tai_from_iso(STOP_TIME))
    DTDB = (TDB1 - TDB0) / MONTHS
    SLOP = 43200.

    CK_LOADED = np.zeros(MONTHS, dtype="bool")      # True if month was loaded
    CK_LIST   = np.empty(MONTHS, dtype="object")    # Kernels needed by month
    CK_DICT   = {}      # Dictionary keyed by filespec returns kernel info
                        # object, but only if loaded.

    SPK_LOADED  = np.zeros(MONTHS, dtype="bool")
    SPK_LIST    = np.empty(MONTHS, dtype="object")
    SPK_DICT    = {}

    loaded_instruments = []

    initialized = False

    ############################################################################

    #===========================================================================
    # initialize
    #===========================================================================
    @staticmethod
    def initialize(ck='reconstructed', planets=None, asof=None,
                   spk='reconstructed', gapfill=True,
                   mst_pck=True, irregulars=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Intialize the Cassini mission internals.

        After the first call, later calls to this function are ignored.

        Input:
            ck,spk      Used to specify which C and SPK kernels are used.:
                        'reconstructed' for the reconstructed kernels (default);
                        'predicted' for the predicted kernels;
                        'none' to allow manual control of the C kernels.
            planets     A list of planets to pass to define_solar_system. None
                        or 0 means all.
            asof        Only use SPICE kernels that existed before this date;
                        None to ignore.
            gapfill     True to include gapfill CKs. False otherwise.
            mst_pck     True to include MST PCKs, which update the rotation
                        models for some of the small moons.
            irregulars  True to include the irregular satellites;
                        False otherwise.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if Cassini.initialized: return

        #---------------------------------------------------------
        # Define some important paths and frames
        #---------------------------------------------------------
        oops.define_solar_system(Cassini.START_TIME, Cassini.STOP_TIME,
                                 asof=asof,
                                 planets=planets,
                                 mst_pck=mst_pck,
                                 irregulars=irregulars)

        ignore = oops.path.SpicePath("CASSINI", "SATURN")

        spicedb.open_db()

        spk = spk.upper()
        if spk == 'NONE':
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
            # This means no SPK will ever be loaded; handling is manual
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
            Cassini.initialize_kernels([], Cassini.SPK_LIST)
            Cassini.SPK_LOADED = np.ones(Cassini.MONTHS, dtype="bool")
        else:
            kernels = spicedb.select_spk(-82, name="CAS-SPK-" + spk,
                                              time=(Cassini.START_TIME,
                                                    Cassini.STOP_TIME),
                                              asof=asof)
            Cassini.initialize_kernels(kernels, Cassini.SPK_LIST)

        ck = ck.upper()
        if ck == 'NONE':
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
            # This means no CK will ever be loaded; handling is manual
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
            Cassini.initialize_kernels([], Cassini.CK_LIST)
            Cassini.CK_LOADED = np.ones(Cassini.MONTHS, dtype="bool")
        else:
            kernels = spicedb.select_ck(-82, name="CAS-CK-" + ck,
                                             time=(Cassini.START_TIME,
                                                   Cassini.STOP_TIME),
                                             asof=asof)
            Cassini.initialize_kernels(kernels, Cassini.CK_LIST)

        #---------------------------------------------------------
        # Load extra kernels if necessary
        #---------------------------------------------------------
        if gapfill and ck not in ('PREDICTED', 'NONE'):
            _ = spicedb.furnish_ck(-82, name="CAS-CK-GAPFILL")

        spicedb.close_db()

        initialized = True
    #===========================================================================



    #===========================================================================
    # reset
    #===========================================================================
    @staticmethod
    def reset():
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Resets the internal parameters. Can be useful for debugging.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        Cassini.loaded_instruments = []

        Cassini.CK_LOADED = np.zeros(Cassini.MONTHS, dtype="bool")
        Cassini.CK_LIST = np.empty(Cassini.MONTHS, dtype="object")
        Cassini.CK_DICT = {}

        Cassini.SPK_LOADED = np.zeros(Cassini.MONTHS, dtype="bool")
        Cassini.SPK_LIST = np.empty(Cassini.MONTHS, dtype="object")
        Cassini.SPK_DICT = {}

        Cassini.initialized = False
    #===========================================================================



    #===========================================================================
    # load_ck
    #===========================================================================
    @staticmethod
    def load_ck(t):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Ensure that the C kernels applicable at or near the given time have
        been furnished. The time can be tai or tdb.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        Cassini.load_kernels(t, t, Cassini.CK_LOADED, Cassini.CK_LIST,
                                   Cassini.CK_DICT)
    #===========================================================================



    #===========================================================================
    # load_cks
    #===========================================================================
    @staticmethod
    def load_cks(t0, t1):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Ensure that all the C kernels applicable near or within the time
        interval tdb0 to tdb1 have been furnished. The time can be tai or tdb.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        Cassini.load_kernels(t0, t1, Cassini.CK_LOADED, Cassini.CK_LIST,
                                     Cassini.CK_DICT)
    #===========================================================================



    #===========================================================================
    # load_spk
    #===========================================================================
    @staticmethod
    def load_spk(t):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Ensure that the SPK kernels applicable at or near the given time have
        been furnished. The time can be tai or tdb.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        Cassini.load_kernels(t, t, Cassini.SPK_LOADED, Cassini.SPK_LIST,
                                   Cassini.SPK_DICT)
    #===========================================================================



    #===========================================================================
    # load_spks
    #===========================================================================
    @staticmethod
    def load_spks(t0, t1):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Ensure that all the SPK kernels applicable near or within the time
        interval tdb0 to tdb1 have been furnished. The time can be tai or tdb.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        Cassini.load_kernels(t0, t1, Cassini.SPK_LOADED, Cassini.SPK_LIST,
                                     Cassini.SPK_DICT)
    #===========================================================================



    #===========================================================================
    # load_kernels
    #===========================================================================
    @staticmethod
    def load_kernels(t0, t1, loaded, lists, kernel_dict):

        #---------------------------------------------------------
        # Find the range of months needed
        #---------------------------------------------------------
        m1 = int((t0 - Cassini.TDB0) // Cassini.DTDB)
        m2 = int((t1 - Cassini.TDB0) // Cassini.DTDB) + 1

        m1 = max(m1, 0)         # ignore time limits outside mission duration
        m2 = min(m2, Cassini.MONTHS - 1)

        #---------------------------------------------------------
        # Load any months not already loaded
        #---------------------------------------------------------
        for m in range(m1, m2+1):
          if not loaded[m]:
            for kernel in lists[m]:
                filespec = kernel.filespec
                if filespec not in kernel_dict:
                    spicedb.furnish_kernels([kernel])
                    kernel_dict[filespec] = kernel
                loaded[m] = True
    #===========================================================================



    ########################################
    # Initialize the kernel lists
    ########################################

    #===========================================================================
    # initialize_kernels
    #===========================================================================
    @staticmethod
    def initialize_kernels(kernels, lists):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        After initialization, lists[m] is a the KernelInfo objects needed
        within the specified month.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        for i in range(Cassini.MONTHS):
            lists[i] = []

        for kernel in kernels:

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            # Find the range of months applicable, extended by 12 hours
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            t0 = cspyce.str2et(kernel.start_time) - Cassini.SLOP
            t1 = cspyce.str2et(kernel.stop_time)  + Cassini.SLOP

            m1 = int((t0 - Cassini.TDB0) // Cassini.DTDB)
            m2 = int((t1 - Cassini.TDB0) // Cassini.DTDB) + 1

            m1 = max(m1, 0)     # ignore time limits outside mission duration
            m2 = min(m2, Cassini.MONTHS - 1)

            #- - - - - - - - - - - - - - - - - - - -
            # Add this kernel to each month's list
            #- - - - - - - - - - - - - - - - - - - -
            for m in range(m1, m2+1):
                lists[m] += [kernel]
    #===========================================================================



    ############################################################################
    # Routines for managing the loading other kernels
    ############################################################################

    #===========================================================================
    # load_instruments
    #===========================================================================
    @staticmethod
    def load_instruments(instruments=[], asof=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Loads the SPICE kernels and defines the basic paths and frames for
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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #---------------------------------------------------------
        # Load the default instruments on the first pass
        #---------------------------------------------------------
        if Cassini.loaded_instruments == []:
            instruments += ["ISS", "VIMS", "CIRS", "UVIS"]

        #---------------------------------------------------------
        # On later calls, return quickly if there's nothing to do
        #---------------------------------------------------------
        if instruments == []: return

        #---------------------------------------------------------
        # Check the formatting of the "as of" date
        #---------------------------------------------------------
        if asof is not None:
            (day, sec) = julian.day_sec_from_iso(asof)
            asof = julian.ymdhms_format_from_day_sec(day, sec)

        #---------------------------------------------------------
        # Furnish instruments and frames
        #---------------------------------------------------------
        spicedb.open_db()
        _ = spicedb.furnish_inst(-82, inst=instruments, asof=asof)
        spicedb.close_db()
    #===========================================================================



    ############################################################################
    # Routines for managing text kernel information
    ############################################################################

    #===========================================================================
    # spice_instrument_kernel
    #===========================================================================
    @staticmethod
    def spice_instrument_kernel(inst, asof=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a dictionary containing the Instrument Kernel information.

        Also furnishes it for use by the SPICE tools.

        Input:
            inst        one of "ISS", "UVIS", "VIMS", "CIRS", etc.
            asof        an optional date in the past, in ISO date or date-time
                        format. If provided, then the information provided will
                        be applicable as of that date. Otherwise, the most
                        recent information is always provided.

        Return:         a tuple containing:
                            the dictionary generated by textkernel.from_file()
                            the name of the kernel.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if asof is not None:
            (day,sec) = julian.day_sec_from_iso(stop_time)
            asof = julian.ymdhms_format_from_day_sec(day, sec)

        spicedb.open_db()
        kernel_info = spicedb.select_inst(-82, types="IK", inst=inst, asof=asof)
        spicedb.furnish_kernels(kernel_info, fast=True)
        spicedb.close_db()

        return (spicedb.as_dict(kernel_info), spicedb.as_names(kernel_info)[0])
    #===========================================================================



    #===========================================================================
    # spice_frames_kernel
    #===========================================================================
    @staticmethod
    def spice_frames_kernel(asof=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a dictionary containing the Cassini Frames Kernel information.

        Also furnishes the kernels for use by the SPICE tools.

        Input:
            asof        an optional date in the past, in ISO date or date-time
                        format. If provided, then the information provided will
                        be applicable as of that date. Otherwise, the most
                        recent information is always provided.

        Return:         a tuple containing:
                            the dictionary generated by textkernel.from_file()
                            an ordered list of the names of the kernels
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if asof is not None:
            (day,sec) = julian.day_sec_from_iso(stop_time)
            asof = julian.ymdhms_format_from_day_sec(day, sec)

        spicedb.open_db()
        kernel_list = spicedb.select_inst(-82, types="FK", asof=asof)
        spicedb.furnish_kernels(kernel_info, fast=True)
        spicedb.close_db()

        return (spicedb.as_dict(kernel_list), spicedb.as_names(kernel_list))
    #===========================================================================



    #===========================================================================
    # used_kernels
    #===========================================================================
    @staticmethod
    def used_kernels(time, inst, return_all_planets=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the list of kernels associated with a Cassini observation at
        a selected range of times.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if return_all_planets:
            bodies = [1, 199, 2, 299, 3, 399, 4, 499, 5, 599, 6, 699,
                      7, 799, 8, 899]
            if time[0] >= TOUR:
                bodies += body.SATURN_MOONS_LOADED
            else:
                bodies += body.JUPITER_MOONS_LOADED
        else:
            if time[0] >= TOUR:
                bodies = [6, 699] + body.SATURN_MOONS_LOADED
            else:
                bodies = [5, 599] + body.JUPITER_MOONS_LOADED

        return spicedb.used_basenames(time=time, inst=inst, sc=-82,
                                      bodies=bodies)
    #===========================================================================


#*******************************************************************************


################################################################################
