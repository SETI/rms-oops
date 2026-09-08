##########################################################################################
# oops/hosts/cassini/__init__.py: Cassini class
#
# Utility functions for managing SPICE kernels while working with Cassini data
# sets.
##########################################################################################

import numpy as np

import julian
import spicedb
import cspyce
import oops

from oops.body import Body

__all__ = ['Cassini']

TOUR = (2003 - 2000) * 365 * 86400      # Rough ET dividing Saturn from Jupiter

##########################################################################################
# Routines for managing the loading of C and SP kernels
##########################################################################################

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

##########################################################################################

class Cassini(object):
    """An instance-free class to hold Cassini-specific parameters.

    Attributes:
        START_TIME (str): The start of the mission as an ISO date.
        STOP_TIME (str): The end of the mission as an ISO date.
        MONTHS (int): The number of equal "months" into which the mission is divided
            for the purpose of loading kernels.
        TDB0 (float): The mission start time in seconds TDB.
        TDB1 (float): The mission stop time in seconds TDB.
        DTDB (float): The duration of one "month" in seconds.
        SLOP (float): The margin, in seconds, by which each month is extended when
            deciding which kernels apply to it.
        CK_LOADED (numpy.ndarray): Boolean array with one flag per month, True if the C
            kernels for that month have been furnished.
        CK_LIST (numpy.ndarray): Object array holding, for each month, the list of
            KernelInfo objects for the C kernels needed within that month.
        CK_DICT (dict[str, KernelInfo]): The furnished C kernels, keyed by filespec.
        SPK_LOADED (numpy.ndarray): Boolean array with one flag per month, True if the
            SP kernels for that month have been furnished.
        SPK_LIST (numpy.ndarray): Object array holding, for each month, the list of
            KernelInfo objects for the SP kernels needed within that month.
        SPK_DICT (dict[str, KernelInfo]): The furnished SP kernels, keyed by filespec.
        loaded_instruments (list[str]): The names of the instruments whose kernels have
            been loaded.
        initialized (bool): True after :meth:`initialize` has been called.
    """

    START_TIME = '1997-10-01'
    STOP_TIME  = '2017-10-01'
    MONTHS = 240        # 20 years * 12 months/year

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

    ######################################################################################

    @staticmethod
    def initialize(ck='reconstructed', planets=None, asof=None,
                   spk='reconstructed', gapfill=True,
                   mst_pck=True, irregulars=True):
        """Intialize the Cassini mission internals.

        After the first call, later calls to this function are ignored.

        Parameters:
            ck (str, optional): The set of C kernels to load, 'reconstructed' or
                'predicted' (case-insensitive); 'none' to load no C kernels
                automatically, leaving their handling to the caller.
            planets (list, optional): A list of planets to pass to
                :meth:`~oops.Body.define_solar_system`. None or 0 means all.
            asof (str, optional): Only use SPICE kernels that existed before this date;
                None to ignore.
            spk (str, optional): The set of SP kernels to load, 'reconstructed' or
                'predicted' (case-insensitive); 'none' to load no SP kernels
                automatically, leaving their handling to the caller.
            gapfill (bool, optional): True to include gapfill CKs. False otherwise.
            mst_pck (bool, optional): True to include MST PCKs, which update the rotation
                models for some of the small moons.
            irregulars (bool, optional): True to include the irregular satellites; False
                otherwise.
        """
        if Cassini.initialized:
            return

        # Define some important paths and frames
        Body.define_solar_system(Cassini.START_TIME, Cassini.STOP_TIME,
                                 asof=asof,
                                 planets=planets,
                                 mst_pck=mst_pck,
                                 irregulars=irregulars)

        _ = oops.path.SpicePath('CASSINI', 'SATURN')

        spicedb.open_db()

        spk = spk.upper()
        if spk == 'NONE':

            # This means no SPK will ever be loaded; handling is manual
            Cassini.initialize_kernels([], Cassini.SPK_LIST)
            Cassini.SPK_LOADED = np.ones(Cassini.MONTHS, dtype='bool')
        else:
            kernels = spicedb.select_spk(-82, name='CAS-SPK-' + spk,
                                              time=(Cassini.START_TIME,
                                                    Cassini.STOP_TIME),
                                              asof=asof)
            Cassini.initialize_kernels(kernels, Cassini.SPK_LIST)

        ck = ck.upper()
        if ck == 'NONE':

            # This means no CK will ever be loaded; handling is manual
            Cassini.initialize_kernels([], Cassini.CK_LIST)
            Cassini.CK_LOADED = np.ones(Cassini.MONTHS, dtype='bool')
        else:
            kernels = spicedb.select_ck(-82, name='CAS-CK-' + ck,
                                             time=(Cassini.START_TIME,
                                                   Cassini.STOP_TIME),
                                             asof=asof)
            Cassini.initialize_kernels(kernels, Cassini.CK_LIST)

        # Load extra kernels if necessary
        if gapfill and ck not in ('PREDICTED', 'NONE'):
            _ = spicedb.furnish_ck(-82, name='CAS-CK-GAPFILL')

        spicedb.close_db()

        Cassini.initialized = True

    @staticmethod
    def reset():
        """Reset the internal parameters.

        Can be useful for debugging.
        """
        Cassini.loaded_instruments = []

        Cassini.CK_LOADED = np.zeros(Cassini.MONTHS, dtype='bool')
        Cassini.CK_LIST = np.empty(Cassini.MONTHS, dtype='object')
        Cassini.CK_DICT = {}

        Cassini.SPK_LOADED = np.zeros(Cassini.MONTHS, dtype='bool')
        Cassini.SPK_LIST = np.empty(Cassini.MONTHS, dtype='object')
        Cassini.SPK_DICT = {}

        Cassini.initialized = False

    @staticmethod
    def load_ck(t):
        """Furnish the C kernels applicable at or near a given time.

        The time can be tai or tdb.

        Parameters:
            t (float): The time in seconds.
        """
        Cassini.load_kernels(t, t, Cassini.CK_LOADED, Cassini.CK_LIST,
                                   Cassini.CK_DICT)

    @staticmethod
    def load_cks(t0, t1):
        """Furnish the C kernels applicable within a time interval.

        Every C kernel applicable near or within the interval `t0` to `t1` is furnished.

        The time can be tai or tdb.

        Parameters:
            t0 (float): The start time in seconds.
            t1 (float): The stop time in seconds.
        """
        Cassini.load_kernels(t0, t1, Cassini.CK_LOADED, Cassini.CK_LIST,
                                     Cassini.CK_DICT)

    @staticmethod
    def load_spk(t):
        """Furnish the SPK kernels applicable at or near a given time.

        The time can be tai or tdb.

        Parameters:
            t (float): The time in seconds.
        """
        Cassini.load_kernels(t, t, Cassini.SPK_LOADED, Cassini.SPK_LIST,
                                   Cassini.SPK_DICT)

    @staticmethod
    def load_spks(t0, t1):
        """Furnish the SPK kernels applicable within a time interval.

        Every SPK kernel applicable near or within the interval `t0` to `t1` is
        furnished.

        The time can be tai or tdb.

        Parameters:
            t0 (float): The start time in seconds.
            t1 (float): The stop time in seconds.
        """
        Cassini.load_kernels(t0, t1, Cassini.SPK_LOADED, Cassini.SPK_LIST,
                                     Cassini.SPK_DICT)

    @staticmethod
    def load_kernels(t0, t1, loaded, lists, kernel_dict):
        """Furnish every kernel in the given lists that applies within a time range.

        Parameters:
            t0 (float): The start time in seconds TDB.
            t1 (float): The stop time in seconds TDB.
            loaded (numpy.ndarray): Boolean array with one flag per month, True if that
                month's kernels have already been furnished; it is updated in place.
            lists (numpy.ndarray): Object array holding, for each month of the mission,
                the list of KernelInfo objects needed within that month.
            kernel_dict (dict[str, KernelInfo]): The furnished KernelInfo objects, keyed
                by filespec; it is updated in place.
        """

        # Find the range of months needed
        m1 = int((t0 - Cassini.TDB0) // Cassini.DTDB)
        m2 = int((t1 - Cassini.TDB0) // Cassini.DTDB) + 1

        m1 = max(m1, 0)         # ignore time limits outside mission duration
        m2 = min(m2, Cassini.MONTHS - 1)

        # Load any months not already loaded
        for m in range(m1, m2+1):
          if not loaded[m]:
            for kernel in lists[m]:
                filespec = kernel.filespec
                if filespec not in kernel_dict:
                    spicedb.furnish_kernels([kernel])
                    kernel_dict[filespec] = kernel
                loaded[m] = True

    ######################################################################################
    # Initialize the kernel lists
    ######################################################################################

    @staticmethod
    def initialize_kernels(kernels, lists):
        """Initialize the monthly lists of KernelInfo objects.

        After initialization, `lists[m]` holds the KernelInfo objects needed within the
        specified month.

        Parameters:
            kernels (list[KernelInfo]): The KernelInfo objects to distribute among the
                months of the mission.
            lists (numpy.ndarray): Object array with one entry per month; each entry is
                replaced by the list of KernelInfo objects that apply within that month,
                extended by :attr:`SLOP` at each end.
        """
        for i in range(Cassini.MONTHS):
            lists[i] = []

        for kernel in kernels:

            # Find the range of months applicable, extended by 12 hours
            t0 = cspyce.str2et(kernel.start_time) - Cassini.SLOP
            t1 = cspyce.str2et(kernel.stop_time)  + Cassini.SLOP

            m1 = int((t0 - Cassini.TDB0) // Cassini.DTDB)
            m2 = int((t1 - Cassini.TDB0) // Cassini.DTDB) + 1

            m1 = max(m1, 0)     # ignore time limits outside mission duration
            m2 = min(m2, Cassini.MONTHS - 1)

            # Add this kernel to each month's list
            for m in range(m1, m2+1):
                lists[m] += [kernel]

    ######################################################################################
    # Routines for managing the loading other kernels
    ######################################################################################

    @staticmethod
    def load_instruments(instruments=[], asof=None):
        """Load the SPICE kernels for the Cassini mission.

        This also defines the basic paths and frames.

        It is generally only to be called once.

        Parameters:
            instruments (list[str], optional): The names of the instruments whose kernels
                to load. On the first call, 'ISS', 'VIMS', 'CIRS' and 'UVIS' are always
                included.
            asof (str, optional): If this specifies a date or date-time in ISO format,
                then only kernels that existed before the specified date are used.
                Otherwise, the most recent versions are always loaded.
        """

        # Load the default instruments on the first pass
        if Cassini.loaded_instruments == []:
            instruments += ['ISS', 'VIMS', 'CIRS', 'UVIS']

        # On later calls, return quickly if there's nothing to do
        if instruments == []:
            return

        # Check the formatting of the "as of" date
        if asof is not None:
            (day, sec) = julian.day_sec_from_iso(asof)
            asof = julian.ymdhms_format_from_day_sec(day, sec)

        # Furnish instruments and frames
        spicedb.open_db()
        _ = spicedb.furnish_inst(-82, inst=instruments, asof=asof)
        spicedb.close_db()

    ######################################################################################
    # Routines for managing text kernel information
    ######################################################################################

### TODO: finish these routines...

    @staticmethod
    def spice_instrument_kernel(inst, asof=None):
        """A dictionary containing the Instrument Kernel information.

        Also furnishes it for use by the SPICE tools.

        Parameters:
            inst (str | list | tuple): One of "ISS", "UVIS", "VIMS", "CIRS", etc.
            asof (str, optional): An optional date in the past, in ISO date or date-time
                format. If provided, then the information provided will be applicable as
                of that date. Otherwise, the most recent information is always provided.

        Returns:
            tuple[dict, str]: The dictionary generated by :func:`textkernel.from_file`
                and the name of the kernel.
        """
        if asof is not None:
            (day,sec) = julian.day_sec_from_iso(stop_time)
            asof = julian.ymdhms_format_from_day_sec(day, sec)

        spicedb.open_db()
        kernel_info = spicedb.select_inst(-82, types='IK', inst=inst, asof=asof)
        spicedb.furnish_kernels(kernel_info, fast=True)
        spicedb.close_db()

        return (spicedb.as_dict(kernel_info), spicedb.as_names(kernel_info)[0])

    @staticmethod
    def spice_frames_kernel(asof=None):
        """A dictionary containing the Cassini Frames Kernel information.

        Also furnishes the kernels for use by the SPICE tools.

        Parameters:
            asof (str, optional): An optional date in the past, in ISO date or date-time
                format. If provided, then the information provided will be applicable as
                of that date. Otherwise, the most recent information is always provided.

        Returns:
            tuple[dict, list[str]]: The dictionary generated by
                :func:`textkernel.from_file` and an ordered list of the names of the
                kernels.
        """
        if asof is not None:
            (day,sec) = julian.day_sec_from_iso(stop_time)
            asof = julian.ymdhms_format_from_day_sec(day, sec)

        spicedb.open_db()
        kernel_list = spicedb.select_inst(-82, types='FK', asof=asof)
        spicedb.furnish_kernels(kernel_info, fast=True)
        spicedb.close_db()

        return (spicedb.as_dict(kernel_list), spicedb.as_names(kernel_list))

    @staticmethod
    def used_kernels(time, inst, return_all_planets=False, ck=True):
        """The kernels associated with a Cassini observation.

        The list covers a selected range of times.

        Parameters:
            time (tuple[float, float]): A (start, stop) tuple of times in seconds TDB.
            inst (str | list | tuple): The instrument name, e.g., 'iss'.
            return_all_planets (bool, optional): Include kernels for all planets not just
                Jupiter or Saturn.
            ck (bool, optional): True (default) to include CK (pointing) kernels; False to
                exclude them, e.g. for an observation whose pointing came from a custom
                cmatrix rather than SPICE, where any furnished CK is unrelated to how the
                observation was actually pointed.

        Returns:
            list[str]: The basenames of the furnished kernels that apply to the
                observation.
        """
        if return_all_planets:
            bodies = [1, 199, 2, 299, 3, 399, 4, 499, 5, 599, 6, 699,
                      7, 799, 8, 899]
            if time[0] >= TOUR:
                bodies += Body.SATURN_MOONS_LOADED
            else:
                bodies += Body.JUPITER_MOONS_LOADED
        else:
            if time[0] >= TOUR:
                bodies = [6, 699] + Body.SATURN_MOONS_LOADED
            else:
                bodies = [5, 599] + Body.JUPITER_MOONS_LOADED

        types = None
        if not ck:
            types = [t for t in spicedb.KERNEL_TYPE_SORT_ORDER if t != 'CK']

        return spicedb.used_basenames(types=types, time=time, inst=inst,
                                      sc=-82, bodies=bodies)

##########################################################################################
