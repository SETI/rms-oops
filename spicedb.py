################################################################################
# spicedb.py
#
# This set of routines handles the selection of SPICE kernels based on various
# criteria related to body, instrument, time frame, etc. It also sorts selected
# kernels into their proper load order.
#
# Deukkwon Yoon & Mark Showalter, PDS Rings Node, SETI Institute, November 2011
################################################################################

import sqlite_db as db      # Here we choose the SQLite3 version
#import mysql_db as db      # Here we choose the MySQL version
import julian
import interval
import cspice
import unittest

# Database description info
SPICE_DB_PATH = "/Library/WebServer/spice.db"
SPICE_FILE_PATH = "/Library/WebServer/SPICE/"

TABLE_NAME = "SPICEDB"
COLUMN_NAMES = ["KERNEL_NAME", "KERNEL_TYPE", "FILESPEC", "START_TIME",
                "STOP_TIME", "RELEASE_DATE", "SPICE_ID", "LOAD_PRIORITY"]

SATURN_BODIES = [9, 399, 699] + range(601,654) + [65035, 65040, 65041, 65045,
                                                  65048, 65050, 65055, 65056]

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

################################################################################
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

    def __cmp__(self, other):
        """The __cmp__ operator is needed for sorting and prioritizing among
        KernelInfo objects."""

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
        if self.spice_id < other.spice_id: return -1
        if self.spice_id > other.spice_id: return +1

        # If all else fails, they're the same
        return 0

################################################################################
# Kernel List Operations
################################################################################

def sort_kernels(kernel_list):
    """Sorts a list of kernels, removing duplicates and putting the rest
    into their proper load order."""

    # Sort into load order
    kernel_list.sort()

    # Remove duplicates
    cleaned_list = kernel_list[0:1]
    for kernel in kernel_list[1:]:
        if kernel != cleaned_list[-1]:
            cleaned_list.append(kernel)

############################################

def remove_overlaps(kernel_list, body, start_time, stop_time):
    """For kernels that have time limits such as CKs and SPKs, this method
    determines which kernels overlap higher-priority kernels and removes
    kernels from the list if they are not required. It returns the filtered
    list in the proper load order.

    Input:
        kernel_list     a sorted list of kernels as returned by sort_kernels,
                        typically after one or more calls to
                            select_kernels("SPK", ...).

        body            an numeric SPICE ID or list of multiple IDs, identifying
                        the complete set of bodies that are required.

        start_time      the start time of the interval of interest, is ISO
                        format "yyyy-hh-mmThh:mm:ss".

        stop_time       the stop time of the interval of interest.

    Return:             a filtered list of kernels, in which unnecessary kernels
                        have been removed. An unnecessary kernel is one whose
                        entire time range is covered by higher-priority kernels.
"""

    # Define the time interval of interest
    (day, sec) = julian.day_sec_type_from_string(start_time)[0:2]
    interval_start_tai = julian.tai_from_day(day) + sec

    (day, sec) = julian.day_sec_type_from_string(stop_time)[0:2]
    interval_stop_tai = julian.tai_from_day(day) + sec

    # Remove overlaps for each body individually
    if type(body) == type(0): body = [body]
    filtered_kernels = []
    for id in body:

        # Create an empty interval
        inter = interval.Interval(interval_start_tai, interval_stop_tai)

        # Insert the kernels for this body, beginning with the lowest priority
        for kernel in kernel_list:
            if kernel.spice_id != id: continue

            (day, sec) = julian.day_sec_type_from_string(kernel.start_time)[0:2]
            kernel_start_tai = julian.tai_from_day(day) + sec

            (day, sec) = julian.day_sec_type_from_string(kernel.stop_time)[0:2]
            kernel_stop_tai = julian.tai_from_day(day) + sec

            inter[(kernel_start_tai,kernel_stop_tai)] = kernel

        # Retrieve the needed kernels in the proper order
        body_list = inter[(interval_start_time, interval_stop_time)]

        # A leading value of None means there is a gap in time coverage
        if body_list[0] is None:
            body_list = body_list[1:]

        # Add this set to the list
        filtered_kernels += body_list

    # Now that we have identified every kernel we need, sort and remove
    # duplicates by erasing the SPICE IDs
    for kernel in filtered_kernels:
        kernel.spice_id = 0

    return sorted_kernels(filtered_kernels)

############################################

def furnish_kernels(kernel_list):
    """Furnishes a sorted list of kernels for use by the CSPICE toolkit. Also
    returns a list of the kernel names."""

    name_list = []
    for kernel in kernel_list:
        cspice.furnsh(SPICE_FILE_PATH + kernel.filespec)
        name = kernel.kernel_name
        if name not in name_list: name_list.append(name)
    
    return name_list

################################################################################
# High-level Database I/O
################################################################################

def open_db(filepath=SPICE_DB_PATH):
    """Opens the SPICE database."""

    db.open(filepath)

def close_db():
    """Opens the SPICE database."""

    db.close()

############################################

def select_kernels(kernel_type, name=None, body=None, time=None, asof=None):
    """Returns a list of KernelInfo objects containing the information returned
    a query performed on the SPICE kernel database.

    Input:
        kernel_type     "SPK", "CK, "IK", "LSK", etc.
        name            a SQL match string for the name of the kernel; use "%"
                        for multiple wildcards and "_" for a single wildcard.
        body            one or more SPICE body IDs.
        time            a tuple consisting of a start and stop time, each
                        expressed as a string in ISO format:
                                "yyyy-mm-ddThh:mm:ss"
        asof            an optional earlier date for which values should be
                        returned. Wherever possible, the kernels selected will
                        have release dates earlier than this date. The date is
                        expressed as a string in ISO format.

    Return:             A list of KernelInfo objects describing the files that
                        match the requirements.
    """

    # Query the database
    sql_string = _sql_query(kernel_type, name, body, time, asof)
    table = db.query(sql_string)

    # If nothing was returned, relax the "asof" constraint and try again
    if len(table) == 0 and asof is not None:
        sql_string = _sql_query(kernel_type, name, body, time, "redo")
        table = db.query(sql_string)

    # If we still have nothing, raise an exception
    if len(table) == 0:
        raise RuntimeError("no results found matching query")

    kernel_info = []
    for row in table:
        kernel_info.append(KernelInfo(row))

    return kernel_list

def _sql_query(kernel_type, name=None, body=None, time=None, asof=None):
    """Internal routien to a query string based on constraints involving the
    kernel type, name, body or bodies, time range, and release date.

    Input:
        kernel_type     "SPK", "CK, "IK", "LSK", etc.
        name            a SQL match string for the name of the kernel; use
                            "%" for multiple wildcards and "_" for a single
                            wildcard.
        body                one or more SPICE body IDs.
        time                a tuple consisting of a start and stop time, each
                            expressed as a string in ISO format:
                                "yyyy-mm-ddThh:mm:ss"
        asof                an optional earlier date for which values should
                            be returned. Wherever possible, the kernels selected
                            will have release dates earlier than this date. The
                            date is expressed as a string in ISO format.

    Return:                 A complete SQL query string.
    """

    query_list  = ["SELECT", KernelInfo.COLUMNS, "FROM SPICEDB\n"]
    query_list += ["WHERE KERNEL_TYPE = '", kernel_type, "'\n"]

    if name != None:
        query_list += ["AND KERNEL_NAME LIKE '", name, "'\n"]

    if body != None:
        if type(body) == type([]) or type(body) == type(()):
            query_list += ["AND SPICE_ID in (", str(body)[2:-2], ")\n"]
        else:
            query_list += ["AND SPICE_ID = '", str(body), "'\n"]

    if time != None:
        query_list += ["AND START_TIME < '", time[1], "'\n"]
        query_list += ["AND STOP_TIME  > '", time[0], "'\n"]

    if asof == "redo":
        query_list += ["ORDER BY RELEASE_DATE ASC\n"]
        query_list += ["LIMIT 1\n"]
    elif asof != None:
        query_list += ["AND RELEASE_DATE <= '", asof, "'\n"]

    elif time == None:
        query_list += ["ORDER BY LOAD_PRIORITY DESC, RELEASE_DATE DESC"]
        query_list += ["LIMIT 1"]

    return " ".join(query_list)

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

            instrument      an optional list of instruments to be used. If the
                            list is empty, C kernels will not be loaded. If one
                            or more instruments are listed, the C kernels and
                            needed Frames kernels will be loaded. Options are
                            the standard mission abbreviations, e.g., "ISS",
                            "VIMS", "CIRS", "UVIS", etc.

        Return:             a list of the names of all the kernels loaded.
        """

        list = []
        spks = []
        cks  = []
        bodies = [-82] + SATURN_BODIES

        # Leapseconds Kernel (LSK)
        list += select_kernels("LSK", asof=asof)

        # While we are at it, lost the LSK file for the Julian Library
        julian.load_kernel(list[0].filespec)

        # Planetary Constants
        list += select_kernels("PCK", asof=asof, name="PCK%")
        list += select_kernels("PCK", asof=asof, name="CPCK_ROCK%")
        list += select_kernels("PCK", asof=asof, name="CPCK_________")

        # Planetary Frames
        list += select_kernels("FK", asof=asof, name="CPK_________")

        # Ephemerides (SP Kernels)
        spks = select_kernels("SPK", asof=asof, time=(start_time, stop_time),
                                     body=bodies)

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

        # Sort and load the various other kernels
        list = sort_kernels(list)
        names = furnish_kernels(list)

        # Sort and load the SPKs
        spks = sort_kernels(spks)
        spks = remove_overlaps(spks, bodies, start_time, stop_time)
        names += furnish_kernels(spks)

        # Sort and load the CKs, if any
        if cks != []:
            cks = sort_kernels(cks)
            cks = remove_overlaps(cks, [-82], start_time, stop_time)
            names += furnish_kernels(cks)

        return names

################################################################################
# UNIT TESTS
################################################################################

class test_select_kernel(unittest.TestCase):

    def runTest(self):

        open_db()


        close_db()

################################################################################
# Perform unit testing if executed from the command line
################################################################################

if __name__ == '__main__':
    unittest.main()

################################################################################
