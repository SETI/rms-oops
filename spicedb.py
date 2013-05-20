################################################################################
# spicedb.py
#
# This set of routines handles the selection of SPICE kernels based on various
# criteria related to body, instrument, time frame, etc. It also sorts selected
# kernels into their proper load order.
#
# Deukkwon Yoon & Mark Showalter, PDS Rings Node, SETI Institute, November 2011
#
# 12/31/11 (MRS) Replaced calls to julian pyparsing routines with calls to much
#   faster ISO parser routines. Added "after" constraint option to queries.
#   Revised remove_overlaps() algorithm to improve performance. Defined
#   set/get_spice_path to check an environment variable if the path has not been
#   defined. Revised open_db() to search for an environment variable if the
#   database name or path is not passed to it. Added furnish_solar_system()
#   function to load a set of kernels sufficient for every moon and planet
#   (including Pluto!)
#
# 1/4/12 (MRS) Minor bugs fixed for cassini and solar system load order.
#
# 1/6/12 (MRS) Added functions as_dict() and as_names().
#
# 1/11/12 (MRS) Fixed bug in select_kernels() using path constraint.
################################################################################

import julian
import textkernel
import interval
import cspice
import unittest
import os

# This is the SQLite3 version of the program
import sqlite_db as db
SPICE_DB_VARNAME = "SPICE_SQLITE_DB_NAME"

# This would be the alternative mechanism for accessing a MySQL database.
# NOT YET IMPLEMENTED!
#import mysql_db as db

TABLE_NAME = "SPICEDB"
COLUMN_NAMES = ["KERNEL_NAME", "KERNEL_TYPE", "FILESPEC", "START_TIME",
                "STOP_TIME", "RELEASE_DATE", "SPICE_ID", "LOAD_PRIORITY"]

# Derived constants
COLUMN_STRING = ", ".join(COLUMN_NAMES)

KERNEL_NAME_INDEX   = COLUMN_NAMES.index("KERNEL_NAME")
KERNEL_TYPE_INDEX   = COLUMN_NAMES.index("KERNEL_TYPE")
FILESPEC_INDEX      = COLUMN_NAMES.index("FILESPEC")
START_TIME_INDEX    = COLUMN_NAMES.index("START_TIME")
STOP_TIME_INDEX     = COLUMN_NAMES.index("STOP_TIME")
RELEASE_DATE_INDEX  = COLUMN_NAMES.index("RELEASE_DATE")
SPICE_ID_INDEX      = COLUMN_NAMES.index("SPICE_ID")
LOAD_PRIORITY_INDEX = COLUMN_NAMES.index("LOAD_PRIORITY")

# For testing at debugging
DEBUG = False   # If true, no files are furnished.
FILE_LIST = []  # If DEBUG, lists the files that would have been furnished.

################################################################################
# Definitions of useful sets of bodies
################################################################################

MARS_ALL_MOONS = range(401,403)

JUPITER_CLASSICAL = range(501,505)
JUPITER_REGULAR   = range(501,506) + range(514,517)
JUPITER_INNER     = [505] + range(514,517)
JUPITER_IRREGULAR = range(506,514) + range(517,550) + [55062, 55063]
JUPITER_ALL_MOONS = range(501,550) + [55062, 55063]

SATURN_CLASSICAL_INNER = range(601,607)     # Mimas through Titan
SATURN_CLASSICAL_OUTER = range(607,609)     # Hyperion, Iapetus
SATURN_CLASSICAL_IRREG = [609]              # Phoebe
SATURN_CLASSICAL  = range(601,610)          # Mimas through Phoebe
SATURN_REGULAR    = range(601,619) + range(632,636) + [649,653] # with Phoebe
SATURN_IRREGULAR  = (range(619,632) + range(636,649) + range(650,653) +
                     [65035, 65040, 65041, 65045, 65048, 65050, 65055, 65056])
SATURN_ALL_MOONS  = (range(601,654) + 
                     [65035, 65040, 65041, 65045, 65048, 65050, 65055, 65056])

URANUS_CLASSICAL  = range(701,706)
URANUS_INNER      = range(706,716) + [725,726,727]
URANUS_REGULAR    = range(701,716) + [725,726,727]
URANUS_IRREGULAR  = range(716,725)
URANUS_ALL_MOONS  = range(701,728)

NEPTUNE_CLASSICAL = range(801,803)
NEPTUNE_INNER     = range(803,809)
NEPTUNE_REGULAR   = [801] + NEPTUNE_INNER
NEPTUNE_IRREGULAR = [802] + range(809,814)
NEPTUNE_ALL_MOONS = range(801,814)

PLUTO_CLASSICAL   = [901]
PLUTO_REGULAR     = range(901,906)
PLUTO_OUTER       = range(902,906)
PLUTO_ALL_MOONS   = range(901,906)

################################################################################
# SPICE file directory tree support
################################################################################

SPICE_PATH = ""

def set_spice_path(spice_path=""):
    """Call to define the directory path to the root of the SPICE file directory
    tree. Call with no argument to reset."""

    global SPICE_PATH

    SPICE_PATH = spice_path

def get_spice_path():
    """Returns the current path to the root of the SPICE file directory tree.
    If the path is undefined, it uses the value of environment variable
    SPICE_PATH."""

    global SPICE_PATH

    if SPICE_PATH == "":
        SPICE_PATH = os.environ["SPICE_PATH"]

    return SPICE_PATH

###############################################################################
# Kernel Information class
################################################################################

class KernelInfo(object):

    def __init__(self, list):
        self.kernel_name   = list[KERNEL_NAME_INDEX]
        self.kernel_type   = list[KERNEL_TYPE_INDEX]
        self.filespec      = list[FILESPEC_INDEX]
        self.start_time    = list[START_TIME_INDEX]
        self.stop_time     = list[STOP_TIME_INDEX]
        self.release_date  = list[RELEASE_DATE_INDEX]
        self.spice_id      = list[SPICE_ID_INDEX]
        self.load_priority = list[LOAD_PRIORITY_INDEX]

    def compare(self, other, ignore_spice_id=False):
        """The compare() operator compares two KernelInfo objects and returns
        -1 if the former should be earlier in load order, 0 if they are equal,
        or +1 if the former should be later in loader order.

        If ignore_spice_id is True, the KernelInfo objects are considered equal
        even if the spice_ids differ, as long as the rest of the information is
        the same."""

        # LSK and SCLK kernels are best loaded first
        if self.kernel_type == "LSK" and other.kernel_type != "LSK": return -1
        if self.kernel_type != "LSK" and other.kernel_type == "LSK": return +1

        if self.kernel_type == "SCLK" and other.kernel_type != "SCLK": return -1
        if self.kernel_type != "SCLK" and other.kernel_type == "SCLK": return +1

        # Other kernel types are organized alphabetically for no particular
        # reason except to keep kernels of the same type together
        if self.kernel_type < other.kernel_type: return -1
        if self.kernel_type > other.kernel_type: return +1

        # First compare load priorities
        if self.load_priority < other.load_priority: return -1
        if self.load_priority > other.load_priority: return +1

        # If load priorities are the same, compare release dates
        if self.release_date < other.release_date: return -1
        if self.release_date > other.release_date: return +1

        # If release dates are the same, load earlier end dates first
        if self.stop_time < other.stop_time: return -1
        if self.stop_time > other.stop_time: return +1

        # If end dates are the same, load later start dates first
        if self.start_time > other.start_time: return -1
        if self.start_time < other.start_time: return +1

        # At this point we might as well just go alphabetical
        if self.kernel_name < other.kernel_name: return -1
        if self.kernel_name > other.kernel_name: return +1

        if self.filespec < other.filespec: return -1
        if self.filespec > other.filespec: return +1

        # Finally, for kernels describing multiple bodies, the order goes by
        # the body ID
        if ignore_spice_id: return 0

        if self.spice_id < other.spice_id: return -1
        if self.spice_id > other.spice_id: return +1

        # If all else fails, they're the same
        return 0

    # Comparison operators, needed for sorting, etc. Note __cmp__ is deprecated.
    def __eq__(self, other):
        if type(self) != type(other): return False
        return self.compare(other, ignore_spice_id=False) == 0

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return self.compare(other, ignore_spice_id=False) <= 0

    def __lt__(self, other):
        return self.compare(other, ignore_spice_id=False) < 0

    def __ge__(self, other):
        return self.compare(other, ignore_spice_id=False) >= 0

    def __gt__(self, other):
        return self.compare(other, ignore_spice_id=False) > 0

    # A test for equivalence that ignores the spice_id
    def same_kernel_as(self, other):
        """Returns True if the two kernels are the same except perhaps for the
        spice_id."""

        return self.compare(other, ignore_spice_id=True) == 0

    def __str__(self):

        return str(self.filespec)

    def __str__(self): return self.__repr__()

    def __repr__(self):

        if self.spice_id is None:
            id = ""
        else:
            id = str(self.spice_id)

        return (self.kernel_name + "/" + self.kernel_type + "/" +
                   self.filespec + "/" + self.start_time + "/" +
                   self.stop_time + "/" + self.release_date + "/" +
                   id + "/" + str(self.load_priority))

####################################
# UNIT TESTS
####################################

class test_KernelInfo(unittest.TestCase):

    # For reference...
    # ["KERNEL_NAME", "KERNEL_TYPE", "FILESPEC", "START_TIME",
    #  "STOP_TIME", "RELEASE_DATE", "SPICE_ID", "LOAD_PRIORITY"]

    def runTest(self):

        # Confirm that LSK always comes first
        info1 = KernelInfo(["NAME", "LSK", "file", "T1", "T2", "T3", 0, 0])
        

        self.assertTrue(info1 == info1)
        self.assertTrue(info1 ==
                KernelInfo(["NAME", "LSK", "file", "T1", "T2", "T3", 0, 0]))

        self.assertTrue(info1 <
                KernelInfo(["NAME", "SCLK", "file", "T1", "T2", "T3", 0, 0]))
        self.assertTrue(info1 <
                KernelInfo(["NAME", "SPK",  "file", "T1", "T2", "T3", 0, 0]))
        self.assertTrue(info1 <
                KernelInfo(["NAME", "CK",   "file", "T1", "T2", "T3", 0, 0]))

        info1 = KernelInfo(["NAME", "SCLK", "file", "T1", "T2", "T3", 0, 0])

        # Confirm that SCLK always comes second
        info1 = KernelInfo(["NAME", "SCLK", "file", "T1", "T2", "T3", 0, 0])

        self.assertTrue(info1 == info1)
        self.assertTrue(info1 ==
                KernelInfo(["NAME", "SCLK", "file", "T1", "T2", "T3", 0, 0]))

        self.assertTrue(info1 >
                KernelInfo(["NAME", "LSK", "file", "T1", "T2", "T3", 0, 0]))
        self.assertTrue(info1 <
                KernelInfo(["NAME", "SPK", "file", "T1", "T2", "T3", 0, 0]))
        self.assertTrue(info1 <
                KernelInfo(["NAME", "CK",  "file", "T1", "T2", "T3", 0, 0]))

        # Confirm that other types sort alphabetically
        info1 = KernelInfo(["NAME", "FK", "file", "T1", "T2", "T3", 0, 0])

        self.assertTrue(info1 == info1)
        self.assertTrue(info1 ==
                KernelInfo(["NAME", "FK", "file", "T1", "T2", "T3", 0, 0]))

        self.assertTrue(info1 >
                KernelInfo(["NAME", "LSK", "file", "T1", "T2", "T3", 0, 0]))
        self.assertTrue(info1 >
                KernelInfo(["NAME", "SCLK", "file", "T1", "T2", "T3", 0, 0]))
        self.assertTrue(info1 >
                KernelInfo(["NAME", "CK",  "file", "T1", "T2", "T3", 0, 0]))
        self.assertTrue(info1 <
                KernelInfo(["NAME", "SPK",  "file", "T1", "T2", "T3", 0, 0]))

        # Check other comparisons
        info1 = KernelInfo(["NAME", "SPK", "file3", "T10", "T20", "T30", 5, 10])

        self.assertTrue(info1 == info1)
        self.assertTrue(info1 ==
            KernelInfo(["NAME", "SPK", "file3", "T10", "T20", "T30", 5, 10]))

        # Body ID
        self.assertTrue(info1 >
            KernelInfo(["NAME", "SPK", "file3", "T10", "T20", "T30", 4, 10]))
        self.assertTrue(info1 <
            KernelInfo(["NAME", "SPK", "file3", "T10", "T20", "T30", 6, 10]))

        # Filespec
        self.assertTrue(info1 >
            KernelInfo(["NAME", "SPK", "file2", "T10", "T20", "T30", 4, 10]))
        self.assertTrue(info1 >
            KernelInfo(["NAME", "SPK", "file2", "T10", "T20", "T30", 5, 10]))
        self.assertTrue(info1 >
            KernelInfo(["NAME", "SPK", "file2", "T10", "T20", "T30", 6, 10]))

        self.assertTrue(info1 <
            KernelInfo(["NAME", "SPK", "file4", "T10", "T20", "T30", 4, 10]))
        self.assertTrue(info1 <
            KernelInfo(["NAME", "SPK", "file4", "T10", "T20", "T30", 5, 10]))
        self.assertTrue(info1 <
            KernelInfo(["NAME", "SPK", "file4", "T10", "T20", "T30", 6, 10]))

        # Kernel name
        self.assertTrue(info1 >
            KernelInfo(["AAAA", "SPK", "file2", "T10", "T20", "T30", 5, 10]))
        self.assertTrue(info1 >
            KernelInfo(["AAAA", "SPK", "file3", "T10", "T20", "T30", 5, 10]))
        self.assertTrue(info1 >
            KernelInfo(["AAAA", "SPK", "file4", "T10", "T20", "T30", 5, 10]))

        self.assertTrue(info1 <
            KernelInfo(["ZZZZ", "SPK", "file2", "T10", "T20", "T30", 5, 10]))
        self.assertTrue(info1 <
            KernelInfo(["ZZZZ", "SPK", "file3", "T10", "T20", "T30", 5, 10]))
        self.assertTrue(info1 <
            KernelInfo(["ZZZZ", "SPK", "file4", "T10", "T20", "T30", 5, 10]))

        # Start time
        self.assertTrue(info1 >
            KernelInfo(["AAAA", "SPK", "file3", "T11", "T20", "T30", 5, 10]))
        self.assertTrue(info1 >
            KernelInfo(["ZZZZ", "SPK", "file3", "T11", "T20", "T30", 5, 10]))

        self.assertTrue(info1 <
            KernelInfo(["AAAA", "SPK", "file3", "T09", "T20", "T30", 5, 10]))
        self.assertTrue(info1 <
            KernelInfo(["ZZZZ", "SPK", "file3", "T09", "T20", "T30", 5, 10]))

        # Stop time
        self.assertTrue(info1 >
            KernelInfo(["AAAA", "SPK", "file3", "T09", "T19", "T30", 5, 10]))
        self.assertTrue(info1 >
            KernelInfo(["ZZZZ", "SPK", "file3", "T09", "T19", "T30", 5, 10]))
        self.assertTrue(info1 >
            KernelInfo(["AAAA", "SPK", "file3", "T11", "T19", "T30", 5, 10]))
        self.assertTrue(info1 >
            KernelInfo(["ZZZZ", "SPK", "file3", "T11", "T19", "T30", 5, 10]))

        self.assertTrue(info1 <
            KernelInfo(["AAAA", "SPK", "file3", "T09", "T21", "T30", 5, 10]))
        self.assertTrue(info1 <
            KernelInfo(["ZZZZ", "SPK", "file3", "T09", "T21", "T30", 5, 10]))
        self.assertTrue(info1 <
            KernelInfo(["AAAA", "SPK", "file3", "T11", "T21", "T30", 5, 10]))
        self.assertTrue(info1 <
            KernelInfo(["ZZZZ", "SPK", "file3", "T11", "T21", "T30", 5, 10]))

        # Release date
        self.assertTrue(info1 >
            KernelInfo(["NAME", "SPK", "file3", "T09", "T20", "T29", 5, 10]))
        self.assertTrue(info1 >
            KernelInfo(["NAME", "SPK", "file3", "T10", "T20", "T29", 5, 10]))
        self.assertTrue(info1 >
            KernelInfo(["NAME", "SPK", "file3", "T11", "T20", "T29", 5, 10]))

        self.assertTrue(info1 <
            KernelInfo(["NAME", "SPK", "file3", "T09", "T20", "T31", 5, 10]))
        self.assertTrue(info1 <
            KernelInfo(["NAME", "SPK", "file3", "T10", "T20", "T31", 5, 10]))
        self.assertTrue(info1 <
            KernelInfo(["NAME", "SPK", "file3", "T11", "T20", "T31", 5, 10]))

        # Load priority
        self.assertTrue(info1 >
            KernelInfo(["NAME", "SPK", "file3", "T10", "T20", "T29", 5,  9]))
        self.assertTrue(info1 >
            KernelInfo(["NAME", "SPK", "file3", "T10", "T20", "T29", 5,  9]))
        self.assertTrue(info1 >
            KernelInfo(["NAME", "SPK", "file3", "T10", "T20", "T29", 5,  9]))

        self.assertTrue(info1 <
            KernelInfo(["NAME", "SPK", "file3", "T10", "T20", "T31", 5, 11]))
        self.assertTrue(info1 <
            KernelInfo(["NAME", "SPK", "file3", "T10", "T20", "T31", 5, 11]))
        self.assertTrue(info1 <
            KernelInfo(["NAME", "SPK", "file3", "T10", "T20", "T31", 5, 11]))

        # Test sorting
        lsk0 = KernelInfo(["NAME", "LSK", "file", "T10", "T20", "T30", 0, 0])
        lsk1 = KernelInfo(["ZZZZ", "LSK", "file", "T10", "T20", "T30", 0, 0])
        lsk2 = KernelInfo(["NAME", "LSK", "file", "T10", "T20", "T40", 0, 0])
        lsk3 = KernelInfo(["NAME", "LSK", "file", "T10", "T20", "T00", 0, 5])

        sclk0 = KernelInfo(["NAME", "SCLK", "file", "T10", "T20", "T30", 0, 0])
        sclk1 = KernelInfo(["ZZZZ", "SCLK", "file", "T10", "T20", "T30", 0, 0])
        sclk2 = KernelInfo(["NAME", "SCLK", "file", "T10", "T20", "T40", 0, 0])
        sclk3 = KernelInfo(["NAME", "SCLK", "file", "T10", "T20", "T00", 0, 5])

        ck0 = KernelInfo(["NAME", "CK", "file0", "T10", "T20", "T30", 3, 0])
        ck1 = KernelInfo(["NAME", "CK", "file0", "T10", "T20", "T30", 4, 0])
        ck2 = KernelInfo(["NAME", "CK", "file1", "T10", "T20", "T30", 3, 0])
        ck3 = KernelInfo(["ZZZZ", "CK", "file0", "T10", "T20", "T30", 3, 0])
        ck4 = KernelInfo(["NAME", "CK", "file0", "T09", "T20", "T30", 3, 0])
        ck5 = KernelInfo(["NAME", "CK", "file0", "T10", "T21", "T30", 3, 0])
        ck6 = KernelInfo(["NAME", "CK", "file0", "T10", "T20", "T31", 3, 0])
        ck7 = KernelInfo(["NAME", "CK", "file0", "T10", "T20", "T30", 3, 1])

        fk0 = KernelInfo(["NAME", "FK", "file0", "T10", "T20", "T30", 3, 0])
        fk1 = KernelInfo(["NAME", "FK", "file0", "T10", "T20", "T30", 4, 0])
        fk2 = KernelInfo(["NAME", "FK", "file1", "T10", "T20", "T30", 3, 0])
        fk3 = KernelInfo(["ZZZZ", "FK", "file0", "T10", "T20", "T30", 3, 0])
        fk4 = KernelInfo(["NAME", "FK", "file0", "T09", "T20", "T30", 3, 0])
        fk5 = KernelInfo(["NAME", "FK", "file0", "T10", "T21", "T30", 3, 0])
        fk6 = KernelInfo(["NAME", "FK", "file0", "T10", "T20", "T31", 3, 0])
        fk7 = KernelInfo(["NAME", "FK", "file0", "T10", "T20", "T30", 3, 1])

        spk0 = KernelInfo(["NAME", "SPK", "file0", "T10", "T20", "T30", 3, 0])
        spk1 = KernelInfo(["NAME", "SPK", "file0", "T10", "T20", "T30", 4, 0])
        spk2 = KernelInfo(["NAME", "SPK", "file1", "T10", "T20", "T30", 3, 0])
        spk3 = KernelInfo(["ZZZZ", "SPK", "file0", "T10", "T20", "T30", 3, 0])
        spk4 = KernelInfo(["NAME", "SPK", "file0", "T09", "T20", "T30", 3, 0])
        spk5 = KernelInfo(["NAME", "SPK", "file0", "T10", "T21", "T30", 3, 0])
        spk6 = KernelInfo(["NAME", "SPK", "file0", "T10", "T20", "T31", 3, 0])
        spk7 = KernelInfo(["NAME", "SPK", "file0", "T10", "T20", "T30", 3, 1])

        random = [fk7, spk0, spk7, lsk2, fk3, ck2, spk6, fk6, spk1, fk1, lsk1,
                  ck5, spk2, fk2, spk4, fk4, sclk0, spk3, ck0, sclk1, lsk3, ck1,
                  sclk2, ck6, spk5, fk5, sclk3, ck3, ck7, fk0, ck4, lsk0]
        random.sort()

        sorted = [lsk0, lsk1, lsk2, lsk3, sclk0, sclk1, sclk2, sclk3,
                  ck0, ck1, ck2, ck3, ck4, ck5, ck6, ck7,
                  fk0, fk1, fk2, fk3, fk4, fk5, fk6, fk7,
                  spk0, spk1, spk2, spk3, spk4, spk5, spk6, spk7]

        self.assertEqual(sorted, random)

################################################################################
# Kernel List Operations
################################################################################

def sort_kernels(kernel_list):
    """Sorts a list of kernels immediately prior to loading. It removes
    duplicate kernels and puts the rest into their proper load order."""

    # Sort into load order
    kernel_list.sort()

    # Copy the first element and set the spice_id to zero
    cleaned_list = kernel_list[0:1]

    # Add each distinct kernel to the list in order
    for kernel in kernel_list[1:]:

        # If it is not already in the list, add it
        if not kernel.same_kernel_as(cleaned_list[-1]):
            cleaned_list.append(kernel)

    return cleaned_list

################################################################################

def remove_overlaps(kernel_list, start_time, stop_time):
    """For kernels that have time limits such as CKs and SPKs, this method
    determines which kernels overlap higher-priority kernels, and removes
    kernels from the list if they are not required. It returns the filtered
    list in the proper load order.

    Input:
        kernel_list     a list of kernels as returned by one or more calls to
                        select_kernels("SPK", ...).

        start_time      the start time of the interval of interest, is ISO
                        format "yyyy-hh-mmThh:mm:ss".

        stop_time       the stop time of the interval of interest.

    Return:             a filtered list of kernels, in which unnecessary kernels
                        have been removed. An unnecessary kernel is one whose
                        entire time range is covered by higher-priority kernels.
    """

    # Sort the kernels
    kernel_list.sort()

    # Construct a list of kernels, one for each body
    body_dict = {}
    for kernel in kernel_list:
        try:
            body_dict[kernel.spice_id] += [kernel,]
        except KeyError:
            body_dict[kernel.spice_id] = [kernel,]

        # Once the list is sorted, we can forget the body id
        kernel.spice_id = None

    # Delete any duplicated lists, because the overlaps will be the same
    for this_id in body_dict.keys():
        for that_id in body_dict.keys():
            if this_id != that_id and body_dict[this_id] == body_dict[that_id]:
                body_dict[this_id] = None
                continue

    for this_id in body_dict.keys():
        if body_dict[this_id] is None:
            del body_dict[this_id]

    # Define the time interval of interest
    interval_start_tai = julian.tai_from_iso(start_time)
    interval_stop_tai  = julian.tai_from_iso(stop_time)

    # Remove overlaps for each body individually
    filtered_kernels = []
    for id in body_dict.keys():

        # Create an empty interval
        inter = interval.Interval(interval_start_tai, interval_stop_tai)

        # Insert the kernels for this body, beginning with the lowest priority
        for kernel in body_dict[id]:

            kernel_start_tai = julian.tai_from_iso(kernel.start_time)
            kernel_stop_tai  = julian.tai_from_iso(kernel.stop_time)

            inter[(kernel_start_tai,kernel_stop_tai)] = kernel

        # Retrieve the needed kernels in the proper order
        body_list = inter[(interval_start_tai, interval_stop_tai)]

        # A leading value of None means there is a gap in time coverage
        if body_list[0] is None:
            body_list = body_list[1:]

        # Add this set to the list
        filtered_kernels += body_list

    return sort_kernels(filtered_kernels)

################################################################################

def furnish_kernels(kernel_list):
    """Furnishes a sorted list of kernels for use by the CSPICE toolkit. Also
    returns a list of the kernel names.
    """

    global DEBUG, FILE_LIST

    if not DEBUG: spice_path = get_spice_path()

    name_list = []
    for kernel in kernel_list:

        if DEBUG:
            FILE_LIST.append(kernel.filespec)
        else:
            cspice.furnsh(os.path.join(spice_path, kernel.filespec))

        name = kernel.kernel_name
        if name not in name_list: name_list.append(name)
    
    return name_list

################################################################################

def as_dict(kernel_list):
    """Returns a dictionary containing all the information in the listed text
    kernels. Binary kernels are ignored.
    """

    spice_path = get_spice_path()

    clear_dict = True       # clear dictionary on the first pass
    for kernel in kernel_list:

        # Check for a text kernel
        ext = os.path.splitext(kernel.filespec)[1].lower()
        if ext[0:2] != ".t": continue

        filespec = os.path.join(spice_path, kernel.filespec)
        result = textkernel.from_file(filespec, clear=clear_dict)

        # On later passes, don't clear the dictionary
        clear_dict = False
    
    return result

################################################################################

def as_names(kernel_list):
    """Returns a list of the names found in the given kernel list.
    """

    name_list = []
    for kernel in kernel_list:
        name = kernel.kernel_name
        if name not in name_list: name_list.append(name)
    
    return name_list

################################################################################
# High-level Database I/O
################################################################################

def open_db(name=None):
    """Opens the SPICE database given its name or file path."""

    if name is None:
        name = os.environ[SPICE_DB_VARNAME]

    db.open(name)

def close_db():
    """Opens the SPICE database."""

    db.close()

################################################################################

def select_kernels(kernel_type, name=None, body=None, time=None, asof=None,
                                after=None, path=None):
    """Returns a list of KernelInfo objects containing the information returned
    from a query performed on the SPICE kernel database.

    Input:
        kernel_type     "SPK", "CK, "IK", "LSK", etc.

        name            a SQL match string for the name of the kernel; use "%"
                        for multiple wildcards and "_" for a single wildcard.

        body            one or more SPICE body IDs.

        time            a tuple consisting of a start and stop time, each
                        expressed as a string in ISO format:
                            "yyyy-mm-ddThh:mm:ss"

        asof            an optional date earlier than today for which values
                        should be returned. Wherever possible, the kernels
                        selected will have release dates earlier than this date.
                        The date is expressed as a string in ISO format.

        after           an optional date such that files originating earlier are
                        not considered. The date is expressed as a string in ISO
                        format.

        path            an optional string that must appear within the file
                        specification path of the kernel.

    Return:             A list of KernelInfo objects describing the files that
                        match the requirements.
    """

    # Query the database
    sql_string = _sql_query(kernel_type, name, body, time, asof, after, path)
    table = db.query(sql_string)

    # If nothing was returned, relax the "asof" and "after" constraints and try
    # again
    if len(table) == 0 and asof is not None:
        sql_string = _sql_query(kernel_type, name, body, time, "redo", after,
                                path)
        table = db.query(sql_string)

    # If we still have nothing, raise an exception
    if len(table) == 0:
        raise RuntimeError("no results found matching query")

    kernel_info = []
    for row in table:
        kernel_info.append(KernelInfo(row))

    return kernel_info

################################################################################

def _sql_query(kernel_type, name=None, body=None, time=None, asof=None,
                            after=None, path=None):
    """This internal routine generates a query string based on constraints
    involving the kernel type, name, body or bodies, time range, and release
    date.

    Input:
        kernel_type     "SPK", "CK, "IK", "LSK", etc.

        name            a SQL match string for the name of the kernel; use "%"
                        for multiple wildcards and "_" for a single wildcard.

        body            one or more SPICE body IDs.

        time            a tuple consisting of a start and stop time, each
                        expressed as a string in ISO format:
                            "yyyy-mm-ddThh:mm:ss"

        asof            an optional date earlier than today for which values
                        should be returned. Wherever possible, the kernels
                        selected will have release dates earlier than this date.
                        The date is expressed as a string in ISO format.

        after           an optional date such that files originating earlier are
                        not considered. The date is expressed as a string in ISO
                        format.

        path            an optional string that must appear within the file
                        specification path of the kernel.

    Return:             A complete SQL query string.
    """

    query_list  = ["SELECT ", COLUMN_STRING, " FROM SPICEDB\n"]
    query_list += ["WHERE KERNEL_TYPE = '", kernel_type, "'\n"]

    if name is not None:
        query_list += ["AND KERNEL_NAME LIKE '", name, "'\n"]

    bodies = 0
    if body is not None:
        if type(body) == type([]) or type(body) == type(()):
            query_list += ["AND SPICE_ID in (", str(body)[1:-1], ")\n"]
            bodies = len(body)
        else:
            query_list += ["AND SPICE_ID = '", str(body), "'\n"]
            bodies = 1

    if time is not None:
        query_list += ["AND START_TIME < '", time[1], "'\n"]
        query_list += ["AND STOP_TIME  > '", time[0], "'\n"]

    if after is not None and asof != "redo":
        query_list += ["AND RELEASE_DATE >= '", after, "'\n"]

    if path is not None:
        query_list += ["AND FILESPEC LIKE '%", path, "%'\n"]

    if asof == "redo":
        query_list += ["ORDER BY RELEASE_DATE ASC\n"]
        query_list += ["LIMIT 1\n"]
    elif asof is not None:
        query_list += ["AND RELEASE_DATE <= '", asof, "'\n"]

    if time is None and bodies <= 1:
        query_list += ["ORDER BY LOAD_PRIORITY DESC, RELEASE_DATE DESC\n"]
        query_list += ["LIMIT 1"]
    else:
        query_list += ["ORDER BY LOAD_PRIORITY ASC, RELEASE_DATE ASC\n"]

    return "".join(query_list)

################################################################################
# Special kernel loader for Cassini
################################################################################

def furnish_cassini_kernels(start_time, stop_time, instrument=None, asof=None):
    """A routine designed to load all needed SPICE kernels for a SPICE
    calculation involving the Cassini spacecraft.

    Input:
        start_time      the start time of the period of interest, in ISO
                        format, "yyyy-mm-ddThh:mm:ss".

        stop_time       the stop time of the period of interest.

        instrument      an optional list of instruments to be used. If the list
                        is empty, C kernels will not be loaded. If one or more
                        instruments are listed, the C kernels and needed Frames
                        kernels will be loaded. Options are the standard mission
                        abbreviations, e.g., "ISS", "VIMS", "CIRS", "UVIS", etc.

        asof            an optional earlier date for which values should be
                        returned. Wherever possible, the kernels selected will
                        have release dates earlier than this date. The date is
                        expressed as a string in ISO format.

    Return:             a list of the names of all the kernels loaded.
    """

    list = []
    spks = []
    cks  = []

    bodies = [699] + SATURN_ALL_MOONS

    # Leapseconds Kernel (LSK)
    list += select_kernels("LSK", asof=asof)

    # While we are at it, load the LSK file for the Julian Library
    if not DEBUG:
       julian.load_from_kernel(os.path.join(get_spice_path(), list[0].filespec))

    # Planetary Constants
    list += select_kernels("PCK", asof=asof, name="PCK%")
    list += select_kernels("PCK", asof=asof, name="CPCK_ROCK%")
    list += select_kernels("PCK", asof=asof, name="CPCK_________")

    # Planetary Frames
    list += select_kernels("FK", asof=asof, name="CAS_ROCKS%")

    # Ephemerides (SP Kernels)
    sat_spks = select_kernels("SPK", asof=asof, time=(start_time, stop_time),
                                     body=bodies, name="SAT%")
    sat_spks = remove_overlaps(sat_spks, start_time, stop_time)

    cas_spks = select_kernels("SPK", asof=asof, time=(start_time, stop_time),
                                     body=-82)

    de_spks = select_kernels("SPK", asof=asof, time=(start_time, stop_time),
                                    body=[10,399,6], name="DE%")
    de_spks.sort()
    spks = sat_spks + cas_spks + de_spks[-1:]

    # The above are sufficient without any instruments
    if instrument is not None:

        # Convert a single character string to a list, for convenience
        if type(instrument) == type(""): instrument = [instrument]

        # Instrument Kernels...
        for id in instrument:
            list += select_kernels("IK", asof=asof, name="CAS_" + id + "%")

        # Spacecraft Frames
        list += select_kernels("FK", asof=asof, name="CAS_V%")
        list += select_kernels("FK", asof=asof, name="CAS_STATUS_V%")

        # Spacecraft Clock
        list += select_kernels("SCLK", asof=asof, body=-82)

        # C (pointing) Kernels
        cks = select_kernels("CK", asof=asof, time=(start_time, stop_time),
                                   body=-82)
        cks = remove_overlaps(cks, start_time, stop_time)

    if DEBUG: FILE_LIST = []

    # Sort and load the various other kernels
    list = sort_kernels(list)
    names = furnish_kernels(list)

    # Load the SPKs
    names += furnish_kernels(spks)

    # Sort and load the CKs, if any
    if cks != []: names += furnish_kernels(cks)

    return names

################################################################################
# Special kernel loader for every planet and moon
################################################################################

def furnish_solar_system(start_time, stop_time, asof=None):
    """A routine designed to load all the SPK, FK and planetary constants files
    needed for the planets and moons of the Solar System.

    Input:
        start_time      the start time of the period of interest, in ISO
                        format, "yyyy-mm-ddThh:mm:ss".

        stop_time       the stop time of the period of interest.

        asof            an optional earlier date for which values should be
                        returned. Wherever possible, the kernels selected will
                        have release dates earlier than this date. The date is
                        expressed as a string in ISO format.

    Return:             a list of the names of all the kernels loaded.
    """

    list = []
    spks = []

    # Leapseconds Kernel (LSK)
    list += select_kernels("LSK", asof=asof)

    # While we are at it, load the LSK file for the Julian Library
    if not DEBUG:
       julian.load_from_kernel(os.path.join(get_spice_path(), list[0].filespec))

    # Planetary Constants, including the latest from Cassini
    list += select_kernels("PCK", asof=asof, name="PCK%")
    list += select_kernels("PCK", asof=asof, name="CPCK_ROCK%")
    list += select_kernels("PCK", asof=asof, name="CPCK_________")

    # Planetary Frames, including Cassini
    list += select_kernels("FK", asof=asof, name="CAS_ROCKS%")

    # Ephemerides (SP Kernels)
    mar_spks = select_kernels("SPK", asof=asof, time=(start_time, stop_time),
                                     body=MARS_ALL_MOONS, name="MAR%")
    mar_spks = remove_overlaps(mar_spks, start_time, stop_time)

    jup_spks = select_kernels("SPK", asof=asof, time=(start_time, stop_time),
                                     body=JUPITER_ALL_MOONS, name="JUP%")
    jup_spks = remove_overlaps(jup_spks, start_time, stop_time)

    sat_spks = select_kernels("SPK", asof=asof, time=(start_time, stop_time),
                                     body=SATURN_ALL_MOONS, name="SAT%")
    sat_spks = remove_overlaps(sat_spks, start_time, stop_time)

    ura_spks = select_kernels("SPK", asof=asof, time=(start_time, stop_time),
                                     body=URANUS_ALL_MOONS, name="URA%")
    ura_spks = remove_overlaps(ura_spks, start_time, stop_time)

    nep_spks = select_kernels("SPK", asof=asof, time=(start_time, stop_time),
                                     body=NEPTUNE_ALL_MOONS, name="NEP%")
    nep_spks = remove_overlaps(nep_spks, start_time, stop_time)

    plu_spks = select_kernels("SPK", asof=asof, time=(start_time, stop_time),
                                     body=PLUTO_ALL_MOONS, name="PLU%")
    plu_spks = remove_overlaps(plu_spks, start_time, stop_time)

    de_spks = select_kernels("SPK", asof=asof, time=(start_time, stop_time),
                                    body=[10,199,299,301,399], name="DE%")
    de_spks.sort()

    spks = (mar_spks + jup_spks + sat_spks + ura_spks + nep_spks + plu_spks +
            de_spks[-1:])

    # Load the kernels
    list = sort_kernels(list)
    names = furnish_kernels(list)
    names += furnish_kernels(spks)

    return names

########################################
# UNIT TESTS
########################################

class test_spicedb(unittest.TestCase):

    def runTest(self):

        global DEBUG, FILE_LIST

        ################################
        # sort_kernels()
        ################################

        # Leapseconds should always come first
        info0 = KernelInfo(["LEAPSECONDS", "LSK", "File0.tls",
                "2000-01-01", "2000-01-02", "2000-01-03", None, 100])

        # Spacecraft clock should always come second
        # These kernels are ordered alphabetically
        info1 = KernelInfo(["SCLK82", "SCLK", "sclk-82.tsc",
                "2000-01-01", "2000-01-02", "2003-01-03", -82, 100])

        info2 = KernelInfo(["SCLK99", "SCLK", "sclk-99.tsc",
                "2000-01-01", "2000-01-02", "2003-01-03", -99, 100])

        # CKs come next alphabetically
        # Lowest load priority comes first, even with later release date
        info3 = KernelInfo(["CK-PREDICTED", "CK", "File2.ck",
                "2001-01-01", "2099-01-01", "2005-01-01", -82, 50])

        # Others are loaded in order of increasing end date
        info4 = KernelInfo(["CK-RECONSTRUCTED", "CK", "File3.ck",
                "2001-01-01", "2002-01-01", "2003-01-01", -82, 100])

        info5 = KernelInfo(["CK-RECONSTRUCTED", "CK", "File3.ck",
                "2002-01-01", "2003-01-01", "2004-01-01", -82, 100])

        # Frames kernels
        # Ordered by release date
        info6 = KernelInfo(["FRAMES-V1", "FK", "File5a.fk",
                "0000-01-01", "9999-12-31", "2004-01-01", 0, 100])

        info7 = KernelInfo(["FRAMES-V2", "FK", "File5b.fk",
                "0000-01-01", "9999-12-31", "2005-01-01", 0, 100])

        # Planetary Constants
        info8 = KernelInfo(["PCK", "PCK", "pck.pck",
                "0000-01-01", "9999-12-31", "2003-01-03", 0, 50])

        # SP Kernels
        # A low-priority predict kernel comes first
        info9 = KernelInfo(["SPK_PREDICTED", "SPK", "predict.spk",
                "2000-01-02", "2020-12-31", "2003-01-03", -82, 50])

        info10 = KernelInfo(["SPK-RECONSTRUCTED", "SPK", "recon.spk",
                "2002-01-01", "2005-01-01", "2003-01-03", -82, 100])

        # These are duplicates and will be skipped
        info10a = KernelInfo(["SPK-RECONSTRUCTED", "SPK", "recon.spk",
                "2002-01-01", "2005-01-01", "2003-01-03", 6, 100])

        info10b = KernelInfo(["SPK-RECONSTRUCTED", "SPK", "recon.spk",
                "2002-01-01", "2005-01-01", "2003-01-03", 601, 100])

        info10c = KernelInfo(["SPK-RECONSTRUCTED", "SPK", "recon.spk",
                "2002-01-01", "2005-01-01", "2003-01-03", 602, 100])

        info10d = KernelInfo(["SPK-RECONSTRUCTED", "SPK", "recon.spk",
                "2002-01-01", "2005-01-01", "2003-01-03", 699, 100])

        info10e = KernelInfo(["SPK-RECONSTRUCTED", "SPK", "recon.spk",
                "2002-01-01", "2005-01-01", "2003-01-03", 699, 100])

        # Another SPK, duplicated for three moons
        info11 = KernelInfo(["SAT123", "SPK", "sat123.spk",
                "1950-01-01", "2050-01-02", "2003-01-03", 619, 100])

        info11a = KernelInfo(["SAT123", "SPK", "sat123.spk",
                "1950-01-01", "2050-01-02", "2003-01-03", 635, 100])

        info11b = KernelInfo(["SAT123", "SPK", "sat123.spk",
                "1950-01-01", "2050-01-02", "2003-01-03", 636, 100])

        # Put in random order
        random_list = [info7, info8, info11, info10b, info10, info6, info10a,
                       info2, info4, info10c, info9, info11a, info3, info5,
                       info1, info11b, info0]

        cleaned_list = [info0, info1, info2, info3, info4, info5, info6, info7,
                        info8, info9, info10, info11]

        self.assertEqual(cleaned_list, sort_kernels(random_list))

        ################################
        # remove_overlaps()
        ################################

        start_time = "2000-01-01T00:00:00"
        stop_time  = "2010-01-01T00:00:00"

        info0 = KernelInfo(["0", "SPK", "0000.spk",
                "1950-01-01", "2050-01-01", "2003-01-01", 6, 100])

        info1 = KernelInfo(["1", "SPK", "1111.spk",
                "1950-01-01", "2002-01-01", "2003-01-02", 6, 100])

        info2 = KernelInfo(["2", "SPK", "2222.spk",
                "1950-01-01", "2003-01-01", "2003-02-01", 6, 100])

        info3 = KernelInfo(["3", "SPK", "3333.spk",
                "2002-07-01", "2004-01-01", "2003-03-01", 6, 100])

        info4 = KernelInfo(["4", "SPK", "4444.spk",
                "1950-01-01", "2002-07-01", "2003-04-01", 6, 100])

        info5 = KernelInfo(["5", "SPK", "5555.spk",
                "2004-01-01", "2050-01-01", "2003-05-01", 6, 100])

        info6 = KernelInfo(["6", "SPK", "6666.spk",
                "2004-01-01", "2004-07-01", "2003-06-01", 6, 100])

        info7 = KernelInfo(["7", "SPK", "7777.spk",
                "1950-01-01", "2050-01-01", "2003-07-01", 6, 100])

        info8 = KernelInfo(["8", "SPK", "8888.spk",
                "2004-07-01", "2050-01-01", "2003-08-01", 6, 100])

        body6 = [info0, info1, info2, info3, info4, info5, info6, info7, info8]
        self.assertEqual(remove_overlaps(body6, start_time, stop_time),
                                         [info7, info8])

        body6 = [info0, info1, info2, info3, info4, info5, info6, info7]
        self.assertEqual(remove_overlaps(body6, start_time, stop_time),
                                         [info7])

        body6 = [info0, info1, info2, info3, info4, info5, info6, info8]
        self.assertEqual(remove_overlaps(body6, start_time, stop_time),
                                         [info3, info4, info6, info8])

        body6 = [info0, info1, info2, info3, info4, info5, info6]
        self.assertEqual(remove_overlaps(body6, start_time, stop_time),
                                         [info3, info4, info5, info6])

        body6 = [info0, info1, info2, info3, info4, info5]
        self.assertEqual(remove_overlaps(body6, start_time, stop_time),
                                         [info3, info4, info5])

        body6 = [info0, info1, info2, info3, info4]
        self.assertEqual(remove_overlaps(body6, start_time, stop_time),
                                         [info0, info3, info4])

        body6 = [info0, info1, info2, info3]
        self.assertEqual(remove_overlaps(body6, start_time, stop_time),
                                         [info0, info2, info3])

        body6 = [info0, info1, info2]
        self.assertEqual(remove_overlaps(body6, start_time, stop_time),
                                         [info0, info2])

        body6 = [info0, info1]
        self.assertEqual(remove_overlaps(body6, start_time, stop_time),
                                         [info0, info1])

        body6 = [info0]
        self.assertEqual(remove_overlaps(body6, start_time, stop_time),
                                         [info0])

        ################################
        # select_kernels()
        ################################

        open_db("test_data/SPICE.db")

        self.assertEqual(select_kernels("LSK")[0].kernel_name, "NAIF0009")

        self.assertEqual(select_kernels("PCK")[0].kernel_name, "PCK00010")

        self.assertEqual(select_kernels("CK", body=-82)[0].kernel_name,
                         "CK-RECONSTRUCTED")

        self.assertEqual(select_kernels("CK", body=-82,
            time=("2008-01-01","2008-01-02"))[0].filespec,
            "Cassini/CK-reconstructed/07362_08002ra.bc")

        self.assertEqual(len(select_kernels("CK", body=-82,
            time=("2008-01-01","2008-02-01"))), 7)

        self.assertEqual(select_kernels("SPK", body=-82,
            time=("2010-01-01","2010-01-02"))[-1].kernel_name,
            "SPK-RECONSTRUCTED")

        self.assertEqual(select_kernels("SPK", body=-82, asof="2009-12-01",
            time=("2010-01-01","2010-01-02"))[-1].kernel_name,
            "SPK-PREDICTED")

        self.assertEqual(select_kernels("SPK", body=-82, name="%PREDICT%",
            time=("2010-01-01","2010-01-02"))[0].kernel_name,
            "SPK-PREDICTED")

        self.assertEqual(select_kernels("SPK", name="SAT%",
            time=("2010-01-01","2010-01-02"),
            body=601)[-1].kernel_name,
            "SAT317")

        self.assertEqual(select_kernels("SPK", name="SAT%",
            time=("2010-01-01","2010-01-02"),
            body=601)[-2].kernel_name,
            "SAT288")

        self.assertEqual(select_kernels("SPK", name="SAT%",
            time=("2010-01-01","2010-01-02"),
            body=601)[-3].kernel_name,
            "SAT286")

        self.assertEqual(select_kernels("SPK", name="SAT%",
            time=("2010-01-01","2010-01-02"),
            body=(601, 603, 605, 607))[-1].kernel_name,
            "SAT317")

        self.assertEqual(select_kernels("SPK", name="SAT%",
            time=("2010-01-01","2010-01-02"),
            body=(601, 603, 605, 607, 65056))[-1].kernel_name,
            "SAT341")

        self.assertEqual(len(select_kernels("SPK", name="SAT%",
            time=("2010-01-01","2010-01-02"),
            body=(601, 603, 605, 607, 65056))), 51)

        self.assertEqual(len(select_kernels("SPK", name="SAT%",
            time=("2010-01-01","2010-01-02"),
            body=(601))), 12)

        self.assertEqual(len(select_kernels("SPK", name="SAT%",
            time=("2010-01-01","2010-01-02"),
            body=(601,602))), 24)

        self.assertEqual(len(select_kernels("SPK", name="SAT%",
            time=("2010-01-01","2010-01-02"),
            body=(65056))), 3)

        ################################
        # furnish_cassini_kernels()
        ################################

        DEBUG = True        # Avoid attempting to load kernels

        names1 = furnish_cassini_kernels("2009-01-01", "2009-02-01")
        self.assertEqual(names1,
            ['NAIF0009', 'CAS_ROCKS_V18',
             'CPCK_ROCK_21JAN2011_MERGED', 'CPCK14OCT2011', 'PCK00010',
             'SAT317', 'SAT341', 'SAT342', 'SAT342-ROCKS',
             'SPK-RECONSTRUCTED', 'DE421'])

        names2 = furnish_cassini_kernels("2009-01-01", "2009-02-01", "ISS")
        self.assertEqual(names2,
            ['NAIF0009',
             'CAS00149',
             'CAS_V40', 'CAS_STATUS_V04', 'CAS_ROCKS_V18', 'CAS_ISS_V10',
             'CPCK_ROCK_21JAN2011_MERGED', 'CPCK14OCT2011', 'PCK00010',
             'SAT317', 'SAT341', 'SAT342', 'SAT342-ROCKS',
             'SPK-RECONSTRUCTED', 'DE421', 'CK-RECONSTRUCTED'])

        DEBUG = False

        ################################
        # furnish_solar_system()
        ################################

        DEBUG = True        # Avoid attempting to load kernels

        names1 = furnish_solar_system("1980-01-01", "2010-01-01")
        self.assertEqual(names1,
            ['NAIF0009',
             'CAS_ROCKS_V18', 'CPCK_ROCK_21JAN2011_MERGED',
             'CPCK14OCT2011', 'PCK00010',
             'MAR085',
             'JUP204', 'JUP230', 'JUP230-ROCKS', 'JUP282',
             'SAT317', 'SAT341', 'SAT342', 'SAT342-ROCKS',
             'URA083', 'URA091', 'URA095',
             'NEP077', 'NEP081', 'NEP085',
             'PLU021',
             'DE421'])

        DEBUG = False

        close_db()

################################################################################
# Perform unit testing if executed from the command line
################################################################################

if __name__ == '__main__':
    unittest.main()

################################################################################
