################################################################################
# hosts/galileo/__init__.py: Galileo class
#
# Utility functions for managing SPICE kernels while working with Galileo data
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

from oops.body import Body

TOUR = (2003 - 2000) * 365 * 86400      # Rough ET dividing Saturn from Jupiter

################################################################################
# Routines for managing the loading of C and SP kernels
################################################################################

# Make sure the leap seconds have been loaded
oops.spice.load_leap_seconds()

# We load CK and SPK files on a very rough month-by-month basis. This is simpler
# than a more granular approach involving detailed calendar calculations. We
# divide the period October 18, 1989 to September 21, 2003 up into "months" of
# equal length. Given any TDB, we quickly determine the month within which it
# falls. Each month is associated with a list of kernels that should be loaded
# whenever information is needed about any time within that month +/- 12 hours.
# The kernels needed for a given month only get loaded when they are needed, and
# are only loaded once. For any geometry calculation involving Galileo, a quick
# call to load_ck(time) or load_spk(time) will ensure that the information is
# available.

################################################################################

class Galileo(object):
    """An instance-free class to hold Galileo-specific parameters."""

    START_TIME = '1989-10-18'
    STOP_TIME  = '2003-09-21'
    MONTHS = 167        # 14 years * 12 months/year - 1 month

    TDB0 = julian.tdb_from_tai(julian.tai_from_iso(START_TIME))
    TDB1 = julian.tdb_from_tai(julian.tai_from_iso(STOP_TIME))
    DTDB = (TDB1 - TDB0) / MONTHS
    SLOP = 43200.

    CK_LOADED = np.zeros(MONTHS, dtype='bool')      # True if month was loaded
    CK_LIST   = np.empty(MONTHS, dtype='object')    # Kernels needed by month
    CK_DICT   = {}      # Dictionary keyed by filespec returns kernel info
                        # object, but only if loaded.

    SPK_LOADED  = np.zeros(MONTHS, dtype='bool')
    SPK_LIST    = np.empty(MONTHS, dtype='object')
    SPK_DICT    = {}

    loaded_instruments = []

    initialized = False

    ############################################################################

    @staticmethod
    def initialize(planets=None, asof=None,
                   mst_pck=True, irregulars=True):
        """Intialize the Galileo mission internals.

        After the first call, later calls to this function are ignored.

        Input:
            planets     A list of planets to pass to define_solar_system. None
                        or 0 means all.
            asof        Only use SPICE kernels that existed before this date;
                        None to ignore.
            mst_pck     True to include MST PCKs, which update the rotation
                        models for some of the small moons.
            irregulars  True to include the irregular satellites;
                        False otherwise.
        """

        if Galileo.initialized:
            return

        # Define some important paths and frames
        Body.define_solar_system(Galileo.START_TIME, Galileo.STOP_TIME,
                                 asof=asof,
                                 planets=planets,
                                 mst_pck=mst_pck,
                                 irregulars=irregulars)

        _ = oops.path.SpicePath('GLL', 'JUPITER')

        spicedb.open_db()

        # This means no SPK will ever be loaded; handling is manual
        Galileo.initialize_kernels([], Galileo.SPK_LIST)
        Galileo.SPK_LOADED = np.ones(Galileo.MONTHS, dtype='bool')

        # This means no CK will ever be loaded; handling is manual
        Galileo.initialize_kernels([], Galileo.CK_LIST)
        Galileo.CK_LOADED = np.ones(Galileo.MONTHS, dtype='bool')

        spicedb.close_db()

        Galileo.initialized = True

    #===========================================================================
    @staticmethod
    def reset():
        """Reset the internal parameters.

        Can be useful for debugging.
        """
        Galileo.loaded_instruments = []

        Galileo.CK_LOADED = np.zeros(Galileo.MONTHS, dtype='bool')
        Galileo.CK_LIST = np.empty(Galileo.MONTHS, dtype='object')
        Galileo.CK_DICT = {}

        Galileo.SPK_LOADED = np.zeros(Galileo.MONTHS, dtype='bool')
        Galileo.SPK_LIST = np.empty(Galileo.MONTHS, dtype='object')
        Galileo.SPK_DICT = {}

        Galileo.initialized = False

    #===========================================================================
    @staticmethod
    def load_ck(t):
        """Ensure that the C kernels applicable at or near the given time have
        been furnished.

        The time can be tai or tdb.
        """
        Galileo.load_kernels(t, t, Galileo.CK_LOADED, Galileo.CK_LIST,
                                   Galileo.CK_DICT)

    #===========================================================================
    @staticmethod
    def load_cks(t0, t1):
        """Ensure that all the C kernels applicable near or within the time
        interval tdb0 to tdb1 have been furnished.

        The time can be tai or tdb.
        """
        Galileo.load_kernels(t0, t1, Galileo.CK_LOADED, Galileo.CK_LIST,
                                     Galileo.CK_DICT)

    #===========================================================================
    @staticmethod
    def load_spk(t):
        """Ensure that the SPK kernels applicable at or near the given time have
        been furnished.

        The time can be tai or tdb.
        """
        Galileo.load_kernels(t, t, Galileo.SPK_LOADED, Galileo.SPK_LIST,
                                   Galileo.SPK_DICT)

    #===========================================================================
    @staticmethod
    def load_spks(t0, t1):
        """Ensure that all the SPK kernels applicable near or within the time
        interval tdb0 to tdb1 have been furnished.

        The time can be tai or tdb.
        """
        Galileo.load_kernels(t0, t1, Galileo.SPK_LOADED, Galileo.SPK_LIST,
                                     Galileo.SPK_DICT)

    #===========================================================================
    @staticmethod
    def load_kernels(t0, t1, loaded, lists, kernel_dict):
        from oops.unittester_support import TESTDATA_PARENT_DIRECTORY
        import glob

        kdir = os.path.join(TESTDATA_PARENT_DIRECTORY, '../SPICE/Galileo/')
        gkdir = os.path.join(TESTDATA_PARENT_DIRECTORY, '../SPICE/General/')

        cspyce.furnsh(gkdir + 'LSK/naif0012.tls')
        cspyce.furnsh(kdir + 'SCLK/mk00062a.tsc')
        cspyce.furnsh(kdir + 'IK/gll36001.ti')
        cspyce.furnsh(kdir + 'SPK/de421.bsp')
        cspyce.furnsh(kdir + 'SPK/de432s.bsp')

        for ckfile in glob.glob(kdir + 'CK/*.bc'):
            cspyce.furnsh(ckfile)

        for spkfile in glob.glob(kdir + 'SPK/*.bsp'):
            cspyce.furnsh(spkfile)

        return

    ############################################################################
    # Initialize the kernel lists
    ############################################################################

    @staticmethod
    def initialize_kernels(kernels, lists):
        """After initialization, lists[m] is the KernelInfo objects needed
        within the specified month.
        """
        for i in range(Galileo.MONTHS):
            lists[i] = []

        for kernel in kernels:

            # Find the range of months applicable, extended by 12 hours
            t0 = cspyce.str2et(kernel.start_time) - Galileo.SLOP
            t1 = cspyce.str2et(kernel.stop_time)  + Galileo.SLOP

            m1 = int((t0 - Galileo.TDB0) // Galileo.DTDB)
            m2 = int((t1 - Galileo.TDB0) // Galileo.DTDB) + 1

            m1 = max(m1, 0)     # ignore time limits outside mission duration
            m2 = min(m2, Galileo.MONTHS - 1)

            # Add this kernel to each month's list
            for m in range(m1, m2+1):
                lists[m] += [kernel]

    ############################################################################
    # Routines for managing the loading of other kernels
    ############################################################################

    @staticmethod
    def load_instruments(instruments=[], asof=None):
        """Load the SPICE kernels and define the basic paths and frames for
        the Galileo mission.

        It is generally only to be called once.

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
        if Galileo.loaded_instruments == []:
           instruments += ['SSI']

        # On later calls, return quickly if there's nothing to do
        if instruments == []:
            return

        # Check the formatting of the "as of" date
        if asof is not None:
            (day, sec) = julian.day_sec_from_iso(asof)
            asof = julian.ymdhms_format_from_day_sec(day, sec)

        # Furnish instruments and frames; Note GLL has no frame kernel
        spicedb.open_db()
        _ = spicedb.furnish_inst(-77, inst=instruments, asof=asof)
        spicedb.close_db()

    ############################################################################
    # Routines for managing text kernel information
    ############################################################################

    #===========================================================================
    @staticmethod
    def used_kernels(time, inst, return_all_planets=False):
        """The list of kernels associated with a Galileo observation at a
        selected range of times.
        """
        if return_all_planets:
            bodies = [1, 199, 2, 299, 3, 399, 4, 499, 5, 599, 6, 699,
                      7, 799, 8, 899]
            if time[0] >= TOUR:
                bodies += Body.JUPITER_MOONS_LOADED
            else:
                bodies += Body.JUPITER_MOONS_LOADED
        else:
            if time[0] >= TOUR:
                bodies = [6, 699] + Body.JUPITER_MOONS_LOADED
            else:
                bodies = [5, 599] + Body.JUPITER_MOONS_LOADED

        return spicedb.used_basenames(time=time, inst=inst, sc=-77,
                                      bodies=bodies)

################################################################################
