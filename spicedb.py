################################################################################
# spicedb.py
#
# This set of routines handles the selection of SPICE kernels based on various
# criteria related to body, instrument, time frame, etc. It also sorts selected
# kernels into their proper load order.
################################################################################

from __future__ import division
import os
import datetime
import unittest

import julian
import interval
import textkernel
import cspyce

import sqlite_db as db

# For testing and debugging
DEBUG = False   # If true, no files are furnished.
ABSPATH_LIST = []   # If DEBUG, lists the files that would have been furnished.

IS_OPEN = False
DB_PATH = ''

TRANSLATOR = None   # Optional user-specified function to alter the absolute
                    # paths of SPICE kernels. This can be used to override the
                    # default kernels to be loaded. See set_translator().

################################################################################
# Global variables to track loaded kernels
################################################################################

# Furnished kernel names by type, listed in load order
FURNISHED_NAMES = {
    'CK':   [],
    'FK':   [],
    'IK':   [],
    'LSK':  [],
    'PCK':  [],
    'SCLK': [],
    'SPK':  [],
    'STARS': [],
    'META':  [],
    'UNK':  [],
}

# Furnished kernel file paths and names by type, listed in load order
FURNISHED_ABSPATHS = {
    'CK':   [],
    'FK':   [],
    'IK':   [],
    'LSK':  [],
    'PCK':  [],
    'SCLK': [],
    'SPK':  [],
    'STARS': [],
    'META':  [],
    'UNK':  [],
}

# Furnished file numbers by name.
FURNISHED_FILENOS = {}

# Furnished sets of kernel file info objects, keyed by basename
FURNISHED_INFO = {}

SPICE_PATH = None

################################################################################
# Kernel Information class
################################################################################

TABLE_NAME = "SPICEDB"
COLUMN_NAMES = ["KERNEL_NAME", "KERNEL_VERSION", "KERNEL_TYPE",
                "FILESPEC", "START_TIME", "STOP_TIME", "RELEASE_DATE",
                "SPICE_ID", "LOAD_PRIORITY", "FULL_NAME", "FILE_NO"]

# Derived constants
COLUMN_STRING = ", ".join(COLUMN_NAMES)

KERNEL_NAME_INDEX    = COLUMN_NAMES.index("KERNEL_NAME")
KERNEL_VERSION_INDEX = COLUMN_NAMES.index("KERNEL_VERSION")
KERNEL_TYPE_INDEX    = COLUMN_NAMES.index("KERNEL_TYPE")
FILESPEC_INDEX       = COLUMN_NAMES.index("FILESPEC")
START_TIME_INDEX     = COLUMN_NAMES.index("START_TIME")
STOP_TIME_INDEX      = COLUMN_NAMES.index("STOP_TIME")
RELEASE_DATE_INDEX   = COLUMN_NAMES.index("RELEASE_DATE")
SPICE_ID_INDEX       = COLUMN_NAMES.index("SPICE_ID")
LOAD_PRIORITY_INDEX  = COLUMN_NAMES.index("LOAD_PRIORITY")
FULL_NAME_INDEX      = COLUMN_NAMES.index("FULL_NAME")
FILE_NO_INDEX        = COLUMN_NAMES.index("FILE_NO")

KERNEL_TYPE_SORT_DICT = {'LSK': 0, 'SCLK': 1, 'FK': 2, 'IK': 3, 'PCK': 4,
                          'SPK': 5, 'CK': 6, 'STARS': 7, 'META': 8}
KERNEL_TYPE_SORT_ORDER = ['LSK', 'SCLK', 'FK', 'IK', 'PCK', 'SPK', 'CK',
                          'STARS', 'META']

KERNEL_TYPE_FROM_EXT = {
    '.tls': 'LSK',
    '.tpc': 'PCK',
    '.bpc': 'PCK',
    '.bsp': 'SPK',
    '.tsc': 'SCLK',
    '.tf' : 'FK',
    '.ti' : 'IK',
    '.bc' : 'CK',
    '.bdb': 'STARS',
    '.txt': 'META',
}
class KernelInfo(object):

    def __init__(self, list):
        self.kernel_name    = list[KERNEL_NAME_INDEX]
        self.kernel_version = list[KERNEL_VERSION_INDEX]
        self.kernel_type    = list[KERNEL_TYPE_INDEX]
        self.filespec       = list[FILESPEC_INDEX]
        self.start_time     = list[START_TIME_INDEX]
        self.stop_time      = list[STOP_TIME_INDEX]
        self.release_date   = list[RELEASE_DATE_INDEX]
        self.spice_id       = list[SPICE_ID_INDEX]
        self.load_priority  = list[LOAD_PRIORITY_INDEX]
        self.basename       = os.path.basename(self.filespec)

        if self.start_time:
            self.start_tai  = julian.tai_from_iso(self.start_time)
            self.stop_tai   = julian.tai_from_iso(self.stop_time)
            self.start_tdb  = julian.tdb_from_tai(self.start_tai)
            self.stop_tdb   = julian.tdb_from_tai(self.stop_tai)
        else:
            self.start_tai  = -1.e99
            self.stop_tai   =  1.e99
            self.start_tdb  = -1.e99
            self.stop_tdb   =  1.e99

        if len(list) > FILE_NO_INDEX:
            self.file_no = list[FILE_NO_INDEX]
        else:
            self.file_no = None

    def compare(self, other):
        """Identify which of two kernels has a higher load priority.

        The compare() operator compares two KernelInfo objects and returns
        -1 if the former should be earlier in load order, 0 if they are equal,
        or +1 if the former should be later in loader order.
        """

        # Compare types
        self_type = KERNEL_TYPE_SORT_DICT[self.kernel_type]
        other_type = KERNEL_TYPE_SORT_DICT[other.kernel_type]

        if self_type < other_type: return -1
        if self_type > other_type: return +1

        # Other kernel types are organized alphabetically for no particular
        # reason except to keep kernels of the same type together
        if self.kernel_type < other.kernel_type: return -1
        if self.kernel_type > other.kernel_type: return +1

        # Compare load priorities
        if self.load_priority < other.load_priority: return -1
        if self.load_priority > other.load_priority: return +1

        # Compare release dates
        if self.release_date < other.release_date: return -1
        if self.release_date > other.release_date: return +1

        # Group names alphabetically
        if self.kernel_name < other.kernel_name: return -1
        if self.kernel_name > other.kernel_name: return +1

        # Earlier versions go first
        if self.kernel_version < other.kernel_version: return -1
        if self.kernel_version > other.kernel_version: return +1

        # Earlier file numbers go first
        if self.file_no < other.file_no: return -1
        if self.file_no > other.file_no: return +1

        # Earlier end dates, later starts go first for better chance of override
        if self.stop_time < other.stop_time: return -1
        if self.stop_time > other.stop_time: return +1

        if self.start_time > other.start_time: return -1
        if self.start_time < other.start_time: return +1

        # Organize by file name if appropriate
        if self.filespec < other.filespec: return -1
        if self.filespec > other.filespec: return +1

        # Finally, organize by file name and SPICE ID
        if self.spice_id < other.spice_id: return -1
        if self.spice_id > other.spice_id: return +1

        # If all else fails, they're the same
        return 0

    # Comparison operators, needed for sorting, etc. Note __cmp__ is deprecated.
    def __eq__(self, other):
        if type(self) != type(other): return False
        return self.compare(other) == 0

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return self.compare(other) <= 0

    def __lt__(self, other):
        return self.compare(other) < 0

    def __ge__(self, other):
        return self.compare(other) >= 0

    def __gt__(self, other):
        return self.compare(other) > 0

    def __str__(self): return self.__repr__()

    def __repr__(self):

        if self.spice_id is None:
            id = ""
        else:
            id = str(self.spice_id)

        result = (self.full_name + "|" +
                  self.kernel_type + "|" +
                  self.filespec + "|" +
                  (self.start_time or '') + "|" +
                  (self.stop_time  or '') + "|" +
                  (self.release_date or '') + "|" +
                  id + "|" +
                  str(self.load_priority))

        if self.file_no is not None:
            result = result + "[" + str(self.file_no) + "]"

        return result

    @property
    def full_name(self):
        # Append version if present
        if self.kernel_version:

            # Separate name and version by a dash unless version starts with '+'
            if self.kernel_version[0] == '+':
                return self.kernel_name + self.kernel_version[1:]
            else:
                return self.kernel_name + '-' + self.kernel_version

        # Otherwise it's just the name
        else:
            return self.kernel_name

    @property
    def timeless(self):
        return (self.start_time is None and self.stop_time is None)

def kernels_from_filespec(filespec, name=None, version=None, release=None,
                                    priority=100):
    """Fill in kernel info as well as possible from a file path."""

    # Search in the database first
    basename = os.path.basename(filespec)
    try:
        if db_is_open():
            return select_by_filespec(basename, time=None)
        else:
            open_db()
            kernels = select_by_filespec(basename, time=None)
            close_db()
            return kernels

    except ValueError:
        pass

    if name is None:
        (name, ext) = os.path.splitext(basename)
        name = name.upper()
    else:
        ext = os.path.splitext(basename)[1]

    ext = ext.lower()

    if version is None:
        version = 'V1'

    full_name = name + '-' + version

    if release is None:
        today = datetime.datetime.today()
        release = '%4d-%02d-%02d' % (today.year, today.month, today.day)

    kernels = []

    # Get info about a CK
    try:
        spice_ids = cspyce.ckobj(filespec)
        for spice_id in spice_ids:
            spice_id = int(spice_id)

            if spice_id < -999:
                body_id = spice_id // 1000
            else:
                body_id = spice_id

            coverages = cspyce.ckcov(filespec, spice_id,
                                     False, 'SEGMENT', 1., 'TDB')
            for (start_tdb, stop_tdb) in coverages:
                start_time = julian.iso_from_tai(julian.tai_from_tdb(start_tdb))
                stop_time  = julian.iso_from_tai(julian.tai_from_tdb(stop_tdb))

                kernel = KernelInfo([name, version, 'CK', filespec,
                                     start_time, stop_time, release,
                                     body_id, priority, full_name, 1])
                kernels.append(kernel)

        return kernels

    except RuntimeError:
        pass

    # Get info about an SPK
    try:
        spice_ids = cspyce.spkobj(filespec)
        for spice_id in spice_ids:
            spice_id = int(spice_id)

            coverages = cspyce.spkcov(filespec, spice_id)
            for (start_tdb, stop_tdb) in coverages:
                start_time = julian.iso_from_tai(julian.tai_from_tdb(start_tdb))
                stop_time  = julian.iso_from_tai(julian.tai_from_tdb(stop_tdb))

                kernel = KernelInfo([name, version, 'SPK', filespec,
                                     start_time, stop_time, release,
                                     spice_id, priority, full_name, 1])
                kernels.append(kernel)

        return kernels

    except RuntimeError:
        pass

    ktype = KERNEL_TYPE_FROM_EXT.get(ext, 'UNK')

    return [KernelInfo([name, version, ktype, filespec, None, None, release,
                        None, priority, full_name, 1])]

################################################################################
# Kernel List Manipulations
################################################################################

def _sort_kernels(kernel_list):
    """Sort a list of KernelInfo objects immediately prior to loading.

    Input:
        kernel_list a list of KernelInfo objects.

    Return:         a new list in which duplicates are removed and the rest are
                    sorted into their proper load order.
    """

    # Sort kernels into load order
    kernel_list.sort()

    # Delete kernels that are no longer needed
    namekeys = []           # ordered list of kernel (name,version)
    bodies_by_name = {}     # dict of bodies vs. kernel (name,version)
    timeless_by_name = {}   # dict of timeless state vs. (name,version)

    # For each kernel...
    for kernel in kernel_list:
        spice_id = kernel.spice_id
        namekey = (kernel.kernel_name, kernel.kernel_version)
        timeless_by_name[namekey] = kernel.timeless

        # Accumulate kernel names in load order and bodies per kernel name
        if namekey in namekeys:
            i = namekeys.index(namekey)
            del namekeys[i]
            namekeys.append(namekey)
            bodies_by_name[namekey] |= {spice_id}
        else:
            namekeys.append(namekey)
            bodies_by_name[namekey] = {spice_id}

    # Delete SPICE IDs that appear in later versions of timeless kernels
    for j in range(len(namekeys)):
        namekey = namekeys[j]
        if not timeless_by_name[namekey]: continue
        if bodies_by_name[namekey] == {None}: continue

        for k in range(j+1,len(namekeys)):
            if namekey[0] == namekeys[k][0]:
                bodies_by_name[namekey] -= bodies_by_name[namekeys[k]]

    # Delete kernels that are no longer needed
    for j in range(len(namekeys)-1, -1, -1):
        if len(bodies_by_name[namekey]) == 0:
            del namekeys[j]

    # Remove kernels that are still used but identical except for the SPICE_ID
    filtered_list = []
    for kernel in kernel_list:
        namekey = (kernel.kernel_name, kernel.kernel_version)
        if namekey not in namekeys: continue

        if kernel.spice_id not in bodies_by_name[namekey]: continue

        for (k,filtered) in enumerate(filtered_list):
            if filtered.filespec == kernel.filespec:
                del filtered_list[k]
                break

        filtered_list.append(kernel)

    return filtered_list

def _remove_overlaps(kernel_list, start_time, stop_time):
    """Filter out kernels completely overridden by higher-priority kernels.

    For kernels that have time limits such as CKs and SPKs, this method
    determines which kernels overlap higher-priority kernels, and removes
    kernels from the list if they are not required. It returns the filtered
    list in the proper load order.

    Input:
        kernel_list a list of KernelInfo objects as returned by one or more
                    calls to select_kernels("SPK", ...), in its intended load
                    order.

        start_time  the start time of the interval of interest, as ISO format
                    "yyyy-hh-mmThh:mm:ss" or as seconds TAI since January 1,
                    2000. None to ignore time limits and just select the most
                    recent kernel(s).

        stop_time   the stop time of the interval of interest. None to ignore
                    time limits.  and just select the most recent kernel(s).

    Return:         A filtered list of kernels, in which unnecessary kernels
                    have been removed. An unnecessary kernel is one whose entire
                    time range is covered by higher-priority kernels.
    """

    # Construct a dictionary of kernel lists, one list for each body
    body_dict = {}
    for kernel in kernel_list:
        if kernel.spice_id not in body_dict:
            body_dict[kernel.spice_id] = []

        body_dict[kernel.spice_id].append(kernel)

    # Sort the kernels in each list
    for kernels in body_dict.values():
        kernels.sort()

    # If time limits are not specified, select the last kernel in each list
    if start_time is None or stop_time is None:
        filtered_kernels = []
        for kernels in body_dict.values():
            full_name = kernels[-1].full_name
            filtered_kernels += [k for k in kernels if k.full_name == full_name]

        return _sort_kernels(filtered_kernels)

    # Define the time interval of interest
    if type(start_time) == str:
        interval_start_tai = julian.tai_from_iso(start_time)
    else:
        interval_start_tai = start_time

    if type(stop_time) == str:
        interval_stop_tai = julian.tai_from_iso(stop_time)
    else:
        interval_stop_tai = start_time

    # Remove overlaps for each body individually
    filtered_kernels = []
    for id in body_dict:

        # Create an empty interval
        inter = interval.Interval(interval_start_tai, interval_stop_tai)

        # Insert the kernels for this body, beginning with the lowest priority
        for kernel in body_dict[id]:
            kernel_start_tai = julian.tai_from_iso(kernel.start_time)
            kernel_stop_tai  = julian.tai_from_iso(kernel.stop_time)

            inter[(kernel_start_tai,kernel_stop_tai)] = kernel

        # Retrieve the needed kernels in the proper order
        interval_kernels = inter[(interval_start_tai, interval_stop_tai)]

        # A leading value of None means there is a gap in time coverage
        if interval_kernels[0] is None:
            interval_kernels = interval_kernels[1:]

        # Add this set to the list
        filtered_kernels += interval_kernels

    return _sort_kernels(filtered_kernels)

def _fileno_str(filenos):
    """Construct a string listing filenos and their ranges inside brackets."""

    # Copy and sort the list
    filenos = list(filenos)
    filenos.sort()

    strlist = ['[', str(filenos[0])]
    k_written = filenos[0]
    k_prev = filenos[0]

    for k in filenos[1:]:

        # Don't write anything till we reach the end of a sequence
        if k == k_prev + 1:
            k_prev = k
            continue

        # Separate single values by commas
        if k_prev == k_written:
            strlist += [',']

        # Use a comma on a list of just two
        elif k_prev == k_written + 1:
            strlist += [',', str(k_prev), ',']

        # Otherwise, use a dash
        else:
            strlist += ['-', str(k_prev), ',']

        strlist += [str(k)]
        k_written = k
        k_prev = k

    if k_prev == k_written:
        pass
    elif k_prev == k_written + 1:
        strlist += [',', str(k_prev)]
    else:
        strlist += ['-', str(k_prev)]

    return ''.join(strlist + [']'])

def _fileno_values(name):
    """Return a kernel name and list of fileno values from a name string."""

    # If there are no file_nos in the name, just return it with an empty list
    if name[-1] != ']':
        return (name, [])

    # Isolate the name and indices
    ibracket = name.index('[')
    indices = name[ibracket+1:-1]
    name = name[:ibracket]

    # Interpret the indices
    filenos = []
    split_by_commas = index.split(',')
    for item in split_by_commas:
        split_by_dash = item.split('-')
        if len(split_by_dash) == 2:
            k0 = int(split_by_dash[0])
            k1 = int(split_by_dash[1])
            for fileno in range(k0,k1+1):
                filenos.append(fileno)
        else:
            filenos.append(str(item))

    return (name, filenos)

################################################################################
# Database Query Support
################################################################################

def _query_kernels(kernel_type, name=None, body=None, time=None, asof=None,
                                after=None, path=None, limit=True, redo=False):
    """Return a list of KernelInfo objects based on the given constraints.

    Input:
        kernel_type "SPK", "CK, "IK", "LSK", etc.

        name        a SQL match string for the name of the kernel; use "%" for
                    multiple wildcards and "_" for a single wildcard.

        body        one or more SPICE body IDs.

        time        a tuple consisting of a start and stop time, each expressed
                    as a string in ISO format, "yyyy-mm-ddThh:mm:ss".
                    Alternatively, times may be given as elapsed seconds TAI
                    since January 1, 2000.

        asof        an optional date earlier than today for which values should
                    be returned. Wherever possible, the kernels selected will
                    have release dates earlier than this date. The date is
                    expressed as a string in ISO format or as a number of
                    seconds TAI elapsed since January 1, 2000.

        after       an optional date such that files originating earlier are not
                    considered. The date is expressed as a string in ISO format
                    or as a number of seconds TAI elapsed since January 1, 2000.

        path        an optional string that must appear within the file
                    specification path of the kernel.

        limit       True to limit the number of returned kernels to one where
                    appropriate; False to return all the matching kernels.

    Return:         A list of KernelInfo objects describing the files that match
                    the requirements.
    """

    # Query the database
    sql_string = _sql_query(kernel_type, name, body, time, asof, after, path,
                                         limit)
    table = db.query(sql_string)

    # If nothing was returned, relax the "asof" and "after" constraints and try
    # again
    if redo and len(table) == 0 and (asof is not None or after is not None):
        sql_string = _sql_query(kernel_type, name, body, time, None, None,
                                path, limit)
        table = db.query(sql_string)

    # If we still have nothing, raise an exception
    if len(table) == 0:
        raise ValueError("no results found matching query")

    kernel_info = []
    for row in table:
        kernel_info.append(KernelInfo(row))

    return kernel_info

def _sql_query(kernel_type, name=None, body=None, time=None, asof=None,
                            after=None, path=None, limit=True):
    """Generate a query string based on the constraints.

    Input:
        kernel_type "SPK", "CK, "IK", "LSK", etc.

        name        a SQL match string for the name of the kernel; use "%" for
                    multiple wildcards and "_" for a single wildcard.

        body        one or more SPICE body IDs.

        time        a tuple consisting of a start and stop time, each expressed
                    as a string in ISO format "yyyy-mm-ddThh:mm:ss".
                    Alternatively, times may be given as elapsed seconds TAI
                    since January 1, 2000.

        asof        an optional date earlier than today for which values should
                    be returned. Wherever possible, the kernels selected will
                    have release dates earlier than this date. The date is
                    expressed as a string in ISO format or as a number of
                    seconds TAI elapsed since January 1, 2000.

        after       an optional date such that files originating earlier are not
                    considered. The date is expressed as a string in ISO format
                    or as a number of seconds TAI elapsed since January 1, 2000.

        path        an optional string that must appear within the file
                    specification path of the kernel.

        limit       True to limit the number of returned kernels to one where
                    appropriate; False to return all the matching kernels.

    Return:         A complete SQL query string.
    """

    # Begin query
    query_list  = ["SELECT ", COLUMN_STRING, " FROM SPICEDB\n"]
    query_list += ["WHERE KERNEL_TYPE = '", kernel_type, "'\n"]

    # Insert kernel name constraint
    if name is not None:
        query_list += ["AND KERNEL_NAME LIKE '", name, "'\n"]

    # Insert body or bodies
    bodies = 0
    if body is not None:
        if type(body) == int:
            query_list += ["AND SPICE_ID = ", str(body), "\n"]
            bodies = 1
        else:
            bodies = len(body)

            if bodies == 1:
                query_list += ["AND SPICE_ID = ", str(body[0]), "\n"]
            else:
                query_list += ["AND SPICE_ID in (", str(list(body))[1:-1],
                               ")\n"]

    # Insert start and stop times
    if time is None: time = (None, None)

    (time0, time1) = time
    if time0 is not None:
        if type(time0) != str:
            time0 = julian.ymdhms_format_from_tai(time0, sep="T", digits=0,
                                                         suffix="")
        query_list += ["AND STOP_TIME  >= '", time0, "'\n"]

    if time1 is not None:
        if type(time1) != str:
            time1 = julian.ymdhms_format_from_tai(time1, sep="T", digits=0,
                                                         suffix="")

        query_list += ["AND START_TIME <= '", time1, "'\n"]

    # Insert path constraint
    if path is not None:
        path = path.replace('\\', '/')  # Must change Windows file separator
        query_list += ["AND FILESPEC LIKE '%", path, "%'\n"]

    # Insert after constraint except on second pass
    if after is not None:
        if type(after) != str:
            after = julian.ymdhms_format_from_tai(after, sep="T", digits=0,
                                                         suffix="")
        query_list += ["AND RELEASE_DATE >= '", after, "'\n"]

    # Insert 'as of' constraint
    if asof is not None:
        if type(asof) != str:
            asof = julian.ymdhms_format_from_tai(asof, sep="T", digits=0,
                                                       suffix="")
        query_list += ["AND RELEASE_DATE <= '", asof, "'\n"]

    # Return limited or unlimited results
    if limit:
        query_list += ["ORDER BY RELEASE_DATE DESC\n", "LIMIT 1\n"]
    else:
        query_list += ["ORDER BY RELEASE_DATE ASC\n"]

    return "".join(query_list)

def _query_by_name(names, time=None):
    """Return a list of KernelInfo objects based on a name (including version).

    Input:
        names       one or more full kernel names, including versions,
                    optionally indexed by file_no ranges.

        time        a tuple consisting of a start and stop time, each expressed
                    as a string in ISO format, "yyyy-mm-ddThh:mm:ss".
                    Alternatively, times may be given as elapsed seconds TAI
                    since January 1, 2000. Use None to return kernels regardless
                    of the time.

    Return:         A list of KernelInfo objects describing the files that match
                    the requirements.
    """

    # Normalize the input
    if type(names) == str: names = [names]

    # Loop through names...
    kernel_info = []

    for name in names:

        # Query the database
        sql_string = _sql_query_by_name(name, time)
        table = db.query(sql_string)

        # If we have nothing, raise an exception
        if len(table) == 0:
            raise ValueError("no results found matching query")

        for row in table:
            kernel_info.append(KernelInfo(row))

    return kernel_info

def _sql_query_by_name(name, time=None):
    """Generate a query string based on a kernel name.

    Input:
        name        a full kernel name including version, optionally indexed by
                    file_no ranges.

        time        a tuple consisting of a start and stop time, each expressed
                    as a string in ISO format, "yyyy-mm-ddThh:mm:ss".
                    Alternatively, times may be given as elapsed seconds TAI
                    since January 1, 2000. Use None to return kernels regardless
                    of the time.

    Return:         A list of KernelInfo objects describing the files that match
                    the requirements.
    """

    # Begin query
    query_list  = ["SELECT ", COLUMN_STRING, " FROM SPICEDB\n"]

    # Extract file_no ranges if necessary
    if name[-1] == ']':
        ibracket = name.index('[')
        index = name[ibracket+1:-1]
        name = name[:ibracket]

        query_list += ["WHERE FULL_NAME = '", name, "'\n"]

        filenos = []
        split_by_commas = index.split(',')
        for item in split_by_commas:
            split_by_dash = item.split('-')
            if len(split_by_dash) == 2:
                k0 = int(split_by_dash[0])
                k1 = int(split_by_dash[1])
                for fileno in range(k0,k1+1):
                    filenos.append(fileno)
            else:
                filenos.append(int(item))

        query_list += ["AND FILE_NO in (", str(list(filenos))[1:-1], ")\n"]

    else:
        query_list += ["WHERE FULL_NAME = '", name, "'\n"]

    # Insert start and stop times
    if time is None: time = (None, None)

    (time0, time1) = time
    if time0 is not None:
        if type(time0) != str:
            time0 = julian.ymdhms_format_from_tai(time0, sep="T", digits=0,
                                                         suffix="")
        query_list += ["AND STOP_TIME  >= '", time0, "'\n"]

    if time1 is not None:
        if type(time1) != str:
            time1 = julian.ymdhms_format_from_tai(time1, sep="T", digits=0,
                                                         suffix="")

        query_list += ["AND START_TIME <= '", time1, "'\n"]

    query_list += ["ORDER BY LOAD_PRIORITY ASC, RELEASE_DATE ASC\n"]

    return "".join(query_list)

def _query_by_filespec(filespecs, time=None):
    """Return a list of KernelInfo objects based on a filename or pattern.

    Input:
        filespec    one file path or match pattern.

        time        a tuple consisting of a start and stop time, each expressed
                    as a string in ISO format, "yyyy-mm-ddThh:mm:ss".
                    Alternatively, times may be given as elapsed seconds TAI
                    since January 1, 2000. Use None to return kernels regardless
                    of the time.

    Return:         A list of KernelInfo objects describing the files that match
                    the pattern.
    """

    # Normalize the input
    if type(filespecs) == str: filespecs = [filespecs]

    # Loop through names...
    kernel_info = []

    for filespec in filespecs:

        # Query the database
        sql_string = _sql_query_by_filespec(filespec, time)

        table = db.query(sql_string)

        # If we have nothing, raise an exception
        if len(table) == 0:
            if time is not None:        # Maybe it's jut out of time range
                sql_string = _sql_query_by_filespec(filespec)
                table2 = db.query(sql_string)
                if len(table2) > 0: continue

            raise ValueError("no results found matching query")

        for row in table:
            kernel_info.append(KernelInfo(row))

    return kernel_info

def _sql_query_by_filespec(filespec, time=None):
    """Generate a query string based on a kernel name.

    Input:
        filespec    one file path or match pattern.

        time        a tuple consisting of a start and stop time, each expressed
                    as a string in ISO format, "yyyy-mm-ddThh:mm:ss".
                    Alternatively, times may be given as elapsed seconds TAI
                    since January 1, 2000. Use None to return kernels regardless
                    of the time.

    Return:         A list of KernelInfo objects describing the files that match
                    the requirements.
    """

    # Begin query
    query_list  = ["SELECT ", COLUMN_STRING, " FROM SPICEDB\n"]
    query_list += ["WHERE FILESPEC like '%", filespec, "'\n"]

    # Insert start and stop times
    if time is None: time = (None, None)

    (time0, time1) = time
    if time0 is not None:
        if type(time0) != str:
            time0 = julian.ymdhms_format_from_tai(time0, sep="T", digits=0,
                                                         suffix="")
        query_list += ["AND STOP_TIME  >= '", time0, "'\n"]

    if time1 is not None:
        if type(time1) != str:
            time1 = julian.ymdhms_format_from_tai(time1, sep="T", digits=0,
                                                         suffix="")

        query_list += ["AND START_TIME <= '", time1, "'\n"]

    query_list += ["ORDER BY LOAD_PRIORITY ASC, RELEASE_DATE ASC\n"]

    return "".join(query_list)

################################################################################
################################################################################
# Public API
################################################################################
################################################################################

def set_spice_path(spice_path=""):
    """Define the directory path to the root of the SPICE file directory tree.

    Call with no argument to reset the path to its default value."""

    global SPICE_PATH

    SPICE_PATH = spice_path

def get_spice_path():
    """Return the current path to the root of the SPICE file directory tree.

    If the path is undefined, it uses the value of environment variable
    SPICE_PATH.
    """

    global SPICE_PATH

    if SPICE_PATH is None:
        SPICE_PATH = os.environ["SPICE_PATH"]

    return SPICE_PATH

def open_db(name=None):
    """Open the SPICE database given its name or file path.

    If no name is given, the value of the environment variable
    SPICE_SQLITE_DB_NAME is used.
    """

    global IS_OPEN, DB_PATH

    if IS_OPEN: return

    if name is None:
        if DB_PATH:
            name = DB_PATH
        else:
            name = os.environ["SPICE_SQLITE_DB_NAME"]

    db.open(name)
    DB_PATH = name
    IS_OPEN = True

def close_db():
    """Close the SPICE database."""

    global IS_OPEN

    if IS_OPEN:
        db.close()
        IS_OPEN = False

def db_is_open():
    """Return True if SPICE database is currently open."""

    global IS_OPEN

    return IS_OPEN

################################################################################
# Filename translator control
################################################################################

def set_translator(func):
    """Define the translator function."""

    global TRANSLATOR

    if TRANSLATOR and not DEBUG:
        raise RuntimeError('spicedb translator can only be defined once')

    if FURNISHED_INFO and not DEBUG:
        raise RuntimeError('spicedb translator cannot be defined after ' +
                           'kernels have already been loaded.')

    TRANSLATOR = func

################################################################################
# Public API for selecting kernels, returning lists of KernelInfo objects
################################################################################

def select_lsk(asof=None, after=None, redo=True):
    """Return a sorted list of leapseconds kernels.

    Input:
        asof        an optional earlier date for which values should be
                    returned. Wherever possible, the kernels selected will have
                    release dates earlier than this date. The date is expressed
                    as a string in ISO format or as a number of seconds TAI
                    elapsed since January 1, 2000.

        after       an optional date such that files originating earlier are not
                    considered. The date is expressed as a string in ISO format
                    or as a number of seconds TAI elapsed since January 1, 2000.

        redo        True to relax the 'asof' and 'after" constraints if no
                    matching results are found; False to raise a ValueError
                    instead.

    Return:         A sorted list of KernelInfo objects.
    """

    # Search the database
    kernel_list = _query_kernels("LSK", asof=asof, after=after, redo=redo,
                                        limit=True)

    # Load the kernels and return the names
    return _sort_kernels(kernel_list)

def select_pck(bodies=None, name=None, asof=None, after=None, redo=True):
    """Return a sorted list of PCKs for one or more bodies.

    Input:
        bodies      one or more SPICE body IDs; None to load kernels for all
                    planetary bodies.

        name        a SQL match string for the name of the kernel; use "%" for
                    multiple wildcards and "_" for a single wildcard.

        asof        an optional earlier date for which values should be
                    returned. Wherever possible, the kernels selected will have
                    release dates earlier than this date. The date is expressed
                    as a string in ISO format or as a number of seconds TAI
                    elapsed since January 1, 2000.

        after       an optional date such that files originating earlier are not
                    considered. The date is expressed as a string in ISO format
                    or as a number of seconds TAI elapsed since January 1, 2000.

        redo        True to relax the 'asof' and 'after" constraints if no
                    matching results are found; False to raise a ValueError
                    instead.

    Return:         A sorted list of KernelInfo objects.
    """

    # Search database
    kernel_list = _query_kernels("PCK", name=name, body=bodies,
                                        asof=asof, after=after, redo=redo,
                                        limit=False)

    # Sort the kernels and return
    return _sort_kernels(kernel_list)

def select_spk(bodies, name=None, time=None, asof=None, after=None, redo=True):
    """Return a sorted list of SPKs for one or more bodies.

    Input:
        bodies      one or more SPICE body IDs; None to load kernels for all
                    planetary bodies.

        name        a SQL match string for the name of the kernel; use "%" for
                    multiple wildcards and "_" for a single wildcard.

        time        a tuple containing the start and stop times. Each time is
                    expressed in either ISO format "yyyy-mm-ddThh:mm:ss" or as a
                    number of seconds TAI elapsed since January 1, 2000. Use
                    None to load the most recent complete set of kernels
                    regardless of their time limits.

        asof        an optional earlier date for which values should be
                    returned. Wherever possible, the kernels selected will have
                    release dates earlier than this date. The date is expressed
                    as a string in ISO format or as a number of seconds TAI
                    elapsed since January 1, 2000.

        after       an optional date such that files originating earlier are not
                    considered. The date is expressed as a string in ISO format
                    or as a number of seconds TAI elapsed since January 1, 2000.

        redo        True to relax the 'asof' and 'after" constraints if no
                    matching results are found; False to raise a ValueError
                    instead.

    Return:         A sorted list of KernelInfo objects.
    """

    # Normalize the input
    if type(bodies) == int: bodies = [bodies]

    # Select the kernels
    spacecraft_only = True
    kernel_list = []
    for body in bodies:
        if body > 0: spacecraft_only = False
        kernel_list += _query_kernels("SPK", name=name, body=body, time=time,
                                             asof=asof, after=after, redo=redo,
                                             limit=False)

    # Remove kernels with overlapping time limits
    if time is None: time = (None, None)
    kernel_list = _remove_overlaps(kernel_list, time[0], time[1])

    # One DE kernel is always required unless only spacecrafts were selected
    if (not spacecraft_only) and (name is None) and \
       (kernel_list[-1].load_priority < 200): # kludge
        kernel_list += _query_kernels("SPK", name="DE%", time=time,
                                             asof=asof, after=after, redo=redo,
                                             limit=True)

        kernel_list = _remove_overlaps(kernel_list, time[0], time[1])

    # Return the sorted list
    return kernel_list

def select_inst(ids, inst=None, types=None, asof=None, after=None, redo=True):
    """Return a sorted list of IKs, FKs and SCLKs for spacecrafts/instruments.

    Input:
        ids         one or more negative SPICE body IDs for spacecrafts.

        inst        one or more instrument names or abbreviations. None to
                    return kernels for every instrument.

        types       one or more kernel types ("IK", "FK", "SCLK") to return.
                    None to return every kernel type.

        asof        an optional earlier date for which values should be
                    returned. Wherever possible, the kernels selected will have
                    release dates earlier than this date. The date is expressed
                    as a string in ISO format or as a number of seconds TAI
                    elapsed since January 1, 2000.

        after       an optional date such that files originating earlier are not
                    considered. The date is expressed as a string in ISO format
                    or as a number of seconds TAI elapsed since January 1, 2000.

        redo        True to relax the 'asof' and 'after' constraints if no
                    matching results are found; False to raise a ValueError
                    instead.

    Return:         A sorted list of KernelInfo objects.
    """

    # Normalize inputs
    if type(ids) == int: ids = [ids]
    if type(inst) == str: inst = [inst]

    if types is None:
        types = ["SCLK", "FK", "IK"]
    elif type(types) == str:
        types = [types]

    # For each spacecraft...
    kernel_list = []
    for id in ids:

        # Select the spacecraft clock kernels
        if "SCLK" in types:
            kernel_list += _query_kernels("SCLK", body=id,
                                          asof=asof, after=after, redo=redo,
                                          limit=True)

        # Select the frames kernels
        if "FK" in types:
            kernel_list += _query_kernels("FK", body=id,
                                          asof=asof, after=after, redo=redo,
                                          limit=False)

        # Select the instrument kernels
        if "IK" in types:
            if inst is None:
                kernel_list += _query_kernels("IK", body=id,
                                              asof=asof, after=after, redo=redo,
                                              limit=False)
            else:
              for name in inst:
                kernel_list += _query_kernels("IK", name='%'+name+'%', body=id,
                                              asof=asof, after=after, redo=redo,
                                              limit=False)

    # Sort the kernels and return
    return _sort_kernels(kernel_list)

def select_ck(ids, name=None, time=None, asof=None, after=None, redo=True):
    """Return a sorted list of CKs for one or more spacecrafts.

    Input:
        ids         one or more negative SPICE body IDs for spacecrafts.

        name        a SQL match string for the name of the kernel; use "%" for
                    multiple wildcards and "_" for a single wildcard.

        time        a tuple containing the start and stop times. Each time is
                    expressed in either ISO format "yyyy-mm-ddThh:mm:ss" or as a
                    number of seconds TAI elapsed since January 1, 2000. Use
                    None to load a complete set of C kernels.

        asof        an optional earlier date for which values should be
                    returned. Wherever possible, the kernels selected will have
                    release dates earlier than this date. The date is expressed
                    as a string in ISO format or as a number of seconds TAI
                    elapsed since January 1, 2000.

        after       an optional date such that files originating earlier are not
                    considered. The date is expressed as a string in ISO format
                    or as a number of seconds TAI elapsed since January 1, 2000.

        redo        True to relax the 'asof' and 'after" constraints if no
                    matching results are found; False to raise a ValueError
                    instead.

    Return:         A sorted list of KernelInfo objects.
    """

    # Normalize inputs
    if type(ids) == int: ids = [ids]

    # For each spacecraft...
    kernel_list = []
    for id in ids:

        # Select the C kernels
        kernel_list += _query_kernels("CK", name=name, time=time,
                                            body=id, asof=asof, after=after,
                                            limit=False)

    # Remove overlapping kernels and sort
    if time is None: time = ('0001-01-01', '3000-01-01')
    return _remove_overlaps(kernel_list, time[0], time[1])

def select_by_name(names, time=None):
    """Return a list of kernel objects associated with a list of names.

    Input:
        names       a list of kernel names, including version numbers, and
                    optional file_no indices.

        time        an optional tuple containing the start and stop times. Each
                    time is expressed in either ISO format "yyyy-mm-ddThh:mm:ss"
                    or as a number of seconds TAI elapsed since January 1, 2000.
                    Use None to load all the matching kernels.
    """

    # Search database
    kernel_list = _query_by_name(names, time)

    # Sort the kernels
    return _sort_kernels(kernel_list)

def select_by_filespec(filespecs, time=None):
    """Return a list of kernel objects associated with a list of names.

    Input:
        names       A list of file specifications or match patterns. The file
                    specification need not contain the directory path.

        time        an optional tuple containing the start and stop times. Each
                    time is expressed in either ISO format "yyyy-mm-ddThh:mm:ss"
                    or as a number of seconds TAI elapsed since January 1, 2000.
                    Use None to load all the matching kernels.
    """

    # Search database, DO NOT sort!
    return _query_by_filespec(filespecs, time)

################################################################################
# Public API for returning text kernels as dictionaries
################################################################################

def as_dict(kernel_list):
    """Return a dictionary containing the information in  text kernels.

    Binary kernels are ignored.
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
# Public API for furnishing kernels
################################################################################

def furnish_kernels(kernel_list, fast=True):
    """Furnish a pre-sorted list of kernels for use by the cspyce module.

    Input:
        kernel_list a pre-sorted list of one or more KernelInfo objects

        fast        True to skip the loading kernels that have already been
                    loaded. False to unload and load them again, thereby raising
                    their priority.

    Return:         an ordered list of the names, versions and file_nos of the
                    kernels loaded. This can be used to re-load the exact same
                    selection of kernels again at a later date.
    """

    global DEBUG, ABSPATH_LIST
    global FURNISHED_NAMES, FURNISHED_ABSPATHS, FURNISHED_INFO
    global FURNISHED_FILENOS
    global TRANSLATOR

    abspath_list = []
    abspath_types = {}      # returns the kernel type given the file abspath
    name_list = []
    name_types = {}
    fileno_dict = {}

    spice_path = get_spice_path()

    # For each kernel...
    for kernel in kernel_list:

        # Add the full name to the end of the name list
        name = kernel.full_name
        if name not in name_list:
            name_list.append(name)
            name_types[name] = kernel.kernel_type

        # Keep track of file_nos required
        if kernel.file_no is not None:
            if name not in fileno_dict:
                fileno_dict[name] = []

            if kernel.file_no not in fileno_dict[name]:
                fileno_dict[name].append(kernel.file_no)

        # Update the list of files to furnish
        filepaths = kernel.filespec.split(',')
        abspaths = [os.path.join(spice_path, f) for f in filepaths]
        if TRANSLATOR:
            new_abspaths = []
            for oldpath in abspaths:
                newpath = TRANSLATOR(oldpath)
                if newpath:
                    new_abspaths.append(newpath)

            abspaths = new_abspaths

        for abspath in abspaths:

            # Remove the name from earlier in the list if necessary
            if abspath in abspath_list:
                abspath_list.remove(abspath)

            # Always add it at the end
            abspath_list.append(abspath)
            abspath_types[abspath] = kernel.kernel_type     # track kernel types

            # Save the info for each furnished file
            basename = os.path.basename(abspath)
            if basename in FURNISHED_INFO:
                FURNISHED_INFO[basename].add(kernel)
            else:
                FURNISHED_INFO[basename] = set([kernel])

    # Furnish the kernel files...
    if DEBUG:
        ABSPATH_LIST += abspath_list

    else:
        for abspath in abspath_list:
            furnished_list = FURNISHED_ABSPATHS[abspath_types[abspath]]

            # In fast mode, avoid re-furnishing kernels
            already_furnished = (abspath in furnished_list)
            if fast and already_furnished:
                continue

            # Otherwise, unload the kernel if it was already furnished
            if already_furnished:
                furnished_list.remove(abspath)
                cspyce.unload(abspath)

            # Load the kernel
            cspyce.furnsh(abspath)
            furnished_list.append(abspath)

        # Track the kernel names loaded
        for name in name_list:
            furnished_names = FURNISHED_NAMES[name_types[name]]

            if name in furnished_names:
                if fast: continue
                furnished_names.remove(name)

            furnished_names.append(name)

    # Append file number ranges into the names in the list returned
    for (name,filenos) in fileno_dict.iteritems():
        k = name_list.index(name)
        name_list[k] = name + _fileno_str(filenos)

        # Track kernels loaded by file_no
        if not DEBUG:
            if name not in FURNISHED_FILENOS:
                FURNISHED_FILENOS[name] = []

            fileno_list = FURNISHED_FILENOS[name]
            for fileno in filenos:
                if fileno in fileno_list:
                    if fast: continue
                    fileno_list.remove(fileno)

                fileno_list.append(fileno)

    return name_list

def furnish_lsk(asof=None, after=None, redo=True, fast=True):
    """Furnish selected leapseconds kernels and return a list of names.

    Input:
        asof        an optional earlier date for which values should be
                    returned. Wherever possible, the kernels selected will have
                    release dates earlier than this date. The date is expressed
                    as a string in ISO format or as a number of seconds TAI
                    elapsed since January 1, 2000.

        after       an optional date such that files originating earlier are not
                    considered. The date is expressed as a string in ISO format
                    or as a number of seconds TAI elapsed since January 1, 2000.

        redo        True to relax the 'asof' and 'after" constraints if no
                    matching results are found; False to raise a ValueError
                    instead.

        fast        True to skip the loading kernels that have already been
                    loaded. False to unload and load them again, thereby raising
                    their priority.

    Return:         A list of kernel names in load order.
    """

    # Search the database
    kernel_list = select_lsk(asof=asof, after=after, redo=redo)

    # Load the kernels and return the names
    return furnish_kernels(kernel_list, fast=fast)

def furnish_pck(bodies=None, name=None, asof=None, after=None, redo=True,
                fast=True):
    """Furnish selected PCKs for one or more bodies.

    Input:
        bodies      one or more SPICE body IDs; None to load kernels for all
                    planetary bodies.

        name        a SQL match string for the name of the kernel; use "%" for
                    multiple wildcards and "_" for a single wildcard.

        asof        an optional earlier date for which values should be
                    returned. Wherever possible, the kernels selected will have
                    release dates earlier than this date. The date is expressed
                    as a string in ISO format or as a number of seconds TAI
                    elapsed since January 1, 2000.

        after       an optional date such that files originating earlier are not
                    considered. The date is expressed as a string in ISO format
                    or as a number of seconds TAI elapsed since January 1, 2000.

        redo        True to relax the 'asof' and 'after" constraints if no
                    matching results are found; False to raise a ValueError
                    instead.

        fast        True to skip the loading kernels that have already been
                    loaded. False to unload and load them again, thereby raising
                    their priority.

    Return:         A list of kernel names in load order.
    """

    # Search database
    kernel_list = select_pck(bodies=bodies, name=name,
                             asof=asof, after=after, redo=redo)

    # Load the kernels and return the names
    return furnish_kernels(kernel_list, fast=fast)

def furnish_spk(bodies, name=None, time=None, asof=None, after=None, redo=True,
                fast=True):
    """Furnish SPKs for one or more bodies and spacecrafts.

    Input:
        bodies      one or more SPICE body IDs; None to load kernels for all
                    planetary bodies.

        name        a SQL match string for the name of the kernel; use "%" for
                    multiple wildcards and "_" for a single wildcard.

        time        a tuple containing the start and stop times. Each time is
                    expressed in either ISO format "yyyy-mm-ddThh:mm:ss" or as a
                    number of seconds TAI elapsed since January 1, 2000. Use
                    None to load the most recent complete set of kernels
                    regardless of their time limits.

        asof        an optional earlier date for which values should be
                    returned. Wherever possible, the kernels selected will have
                    release dates earlier than this date. The date is expressed
                    as a string in ISO format or as a number of seconds TAI
                    elapsed since January 1, 2000.

        after       an optional date such that files originating earlier are not
                    considered. The date is expressed as a string in ISO format
                    or as a number of seconds TAI elapsed since January 1, 2000.

        redo        True to relax the 'asof' and 'after" constraints if no
                    matching results are found; False to raise a ValueError
                    instead.

        fast        True to skip the loading kernels that have already been
                    loaded. False to unload and load them again, thereby raising
                    their priority.

    Return:         A list of kernel names in load order.
    """

    # Search database
    kernel_list = select_spk(bodies, name=name, time=time, asof=asof,
                             after=after, redo=redo)

    # Furnish the kernels and return the names
    return furnish_kernels(kernel_list, fast=fast)

def furnish_inst(ids, inst=None, types=None, asof=None, after=None, redo=True,
                      fast=True):
    """Furnish IKs, FKs and SCLKs for one or more spacecrafts and instruments.

    Input:
        ids         one or more negative SPICE body IDs for spacecrafts.

        inst        one or more instrument names or abbreviations. None to
                    furnish kernels for every instrument.

        types       one or more kernel types ("IK", "FK", "SCLK") to furnish.
                    None to return every kernel type.

        asof        an optional earlier date for which values should be
                    returned. Wherever possible, the kernels selected will have
                    release dates earlier than this date. The date is expressed
                    as a string in ISO format or as a number of seconds TAI
                    elapsed since January 1, 2000.

        after       an optional date such that files originating earlier are not
                    considered. The date is expressed as a string in ISO format
                    or as a number of seconds TAI elapsed since January 1, 2000.

        redo        True to relax the 'asof' and 'after" constraints if no
                    matching results are found; False to raise a ValueError
                    instead.

    Return:         A list of kernel names in load order.
    """

    # Search database
    kernel_list = select_inst(ids, inst, types, asof, after, redo)

    # Furnish the kernels and return the names
    return furnish_kernels(kernel_list, fast=fast)

def furnish_ck(ids, name=None, time=None, asof=None, after=None, redo=True,
                    fast=True):
    """Furnish CKs for one or more spacecrafts.

    Input:
        ids         one or more negative SPICE body IDs for spacecrafts.

        name        a SQL match string for the name of the kernel; use "%" for
                    multiple wildcards and "_" for a single wildcard.

        time        a tuple containing the start and stop times. Each time is
                    expressed in either ISO format "yyyy-mm-ddThh:mm:ss" or as a
                    number of seconds TAI elapsed since January 1, 2000. Use
                    None to load a complete set of C kernels.

        asof        an optional earlier date for which values should be
                    returned. Wherever possible, the kernels selected will have
                    release dates earlier than this date. The date is expressed
                    as a string in ISO format or as a number of seconds TAI
                    elapsed since January 1, 2000.

        after       an optional date such that files originating earlier are not
                    considered. The date is expressed as a string in ISO format
                    or as a number of seconds TAI elapsed since January 1, 2000.

        redo        True to relax the 'asof' and 'after" constraints if no
                    matching results are found; False to raise a ValueError
                    instead.

        fast        True to skip the loading kernels that have already been
                    loaded. False to unload and load them again, thereby raising
                    their priority.

    Return:         A list of kernel names in load order.
    """

    # Search database
    kernel_list = select_ck(ids, name=name, time=time,
                            asof=asof, after=after, redo=redo)

    # Furnish the kernels and return the names
    return furnish_kernels(kernel_list, fast=fast)

def furnish_by_name(names, time=None, fast=True):
    """Furnish kernels identified by a list of names.

    Input:
        names       a list of kernel names, including version numbers, and
                    optional file_no indices.

        time        an optional tuple containing the start and stop times. Each
                    time is expressed in either ISO format "yyyy-mm-ddThh:mm:ss"
                    or as a number of seconds TAI elapsed since January 1, 2000.
                    Use None to load all the matching kernels.

        fast        True to skip the loading kernels that have already been
                    loaded. False to unload and load them again, thereby raising
                    their priority.

    Return:         A list of kernel names in load order. This will typically
                    match the input names unless different time limits are
                    applied.
    """

    # Search database
    kernel_list = select_by_name(names, time)

    # Furnish the kernels and return the names
    return furnish_kernels(kernel_list, fast=fast)

def furnish_by_metafile(metafile, time=None, asof=None):
    """Furnish kernels identified by the path to a metakernel.

    Input:
        metafile    a file path to a metafile, or the name of a metafile in
                    the SPICE database, or the filespec of a meta kernel in
                    the SPICE database.

        time        a tuple consisting of a start and stop time, each expressed
                    as a string in ISO format, "yyyy-mm-ddThh:mm:ss".
                    Alternatively, times may be given as elapsed seconds TAI
                    since January 1, 2000. Use None to return kernels regardless
                    of the time.

        asof        an optional date earlier than today for which values should
                    be returned. Wherever possible, the kernels selected will
                    have release dates earlier than this date. The date is
                    expressed as a string in ISO format or as a number of
                    seconds TAI elapsed since January 1, 2000.

    Return:         A list of kernel names in load order.
    """

    # Search database
    kernel_names = []
    if not os.path.exists(metafile):
        spice_path = get_spice_path()
        try:
            kernel_list = _query_kernels('META', name=metafile, asof=asof)
            metafile = os.path.join(spice_path, kernel_list[-1].filespec)
            kernel_names = [kernel_list[-1].full_name]
        except ValueError:
            kernel_list = _query_kernels('META', path=metafile, asof=asof)
            metafile = os.path.join(spice_path, kernel_list[-1].filespec)
            kernel_names = [kernel_list[-1].full_name]

    filespecs = textkernel.from_file(metafile)['KERNELS_TO_LOAD']

    kernel_list = select_by_filespec(filespecs, time=time)

    # Furnish the kernels and return the names
    return furnish_kernels(kernel_list, fast=False) + kernel_names

def furnish_by_filepath(filepath):
    """Furnish a file by its full file path. This file need not be in the
    database."""

    kernels = kernels_from_filespec(filepath)
    furnish_kernels(kernels, fast=False)

################################################################################
# Public API for unloading kernels
################################################################################

def unload_by_name(names):
    """Unload kernels based on a list of kernel names."""

    global FURNISHED_ABSPATHS, FURNISHED_NAMES, FURNISHED_INFO
    global FURNISHED_FILENOS

    # Search database
    kernel_list = _query_by_name(names)

    # Sort the kernels
    kernel_list = _sort_kernels(kernel_list)

    # For each kernel...
    spice_path = get_spice_path()
    for kernel in kernel_list:
        key = kernel.kernel_type

        # Remove the kernel files from the dictionary and unload from SPICE
        filespecs = kernel.filespec.split(',')
        abspaths = [os.path.join(spice_path, f) for f in filespecs]
        for abspath in abspaths:
            if abspath in FURNISHED_ABSPATHS[key]:
                FURNISHED_ABSPATHS[key].remove(abspath)
                del FURNISHED_INFO[os.path.basename(abspath)]
                cspyce.unload(abspath)

        # Delete the file_no from the list
        name = kernel.full_name
        if name in FURNISHED_FILENOS:
            fileno_list = FURNISHED_FILENOS[name]
            if kernel.file_no in fileno_list:
                fileno_list.remove(kernel.file_no)

                if len(fileno_list) == 0:
                    del FURNISHED_FILENOS[name]

        # Delete the kernel name from the dictionaries if there a no other files
        if name not in FURNISHED_FILENOS:
            furnished_list = FURNISHED_NAMES[key]
            if name in furnished_list:
                furnished_list.remove(name)

    return

def unload_by_type(types=None):
    """Unload all the kernels of one or more specified types."""

    global FURNISHED_ABSPATHS, FURNISHED_NAMES, FURNISHED_INFO
    global FURNISHED_FILENOS, KERNEL_TYPE_SORT_ORDER

    # Normalize input
    if types is None or types == []:
        types = KERNEL_TYPE_SORT_ORDER
    elif type(types) == str:
        types = [types]

    spice_path = get_spice_path()

    # For each selected type...
    for key in types:

        # Unload each file from SPICE
        abspath_list = FURNISHED_ABSPATHS[key]
        for file in abspath_list:
            cspyce.unload(os.path.join(spice_path, file))
            del FURNISHED_INFO[os.path.basename(file)]

        # Delete the file list from the dictionary
        FURNISHED_ABSPATHS[key] = []

        # Delete the file_no list if necessary
        name_list = FURNISHED_NAMES[key]
        for name in name_list:
            if name in FURNISHED_FILENOS:
                del FURNISHED_FILENOS[name]

        # Delete the name list from the dictionary
        FURNISHED_NAMES[key] = []

    return

def unload_by_filepath(filepath):
    """Unload a file by its full file path. This file need not be in the
    database."""

    kernels = kernels_from_filespec(filepath)
    name = kernels[0].full_name
    ktype = kernels[0].kernel_type
    basename = os.path.basename(filepath)

    if name in FURNISHED_NAMES[ktype]:
        FURNISHED_NAMES[ktype].remove(name)

    if filepath in FURNISHED_ABSPATHS[ktype]:
        FURNISHED_ABSPATHS[ktype].remove(filepath)

    del FURNISHED_INFO[basename]

    if name in FURNISHED_FILENOS:
        del FURNISHED_FILENOS[name]

################################################################################
# Public API for names of kernels
################################################################################

def as_names(kernels):
    """Return a list of names identifying a list of KernelInfo objects."""

    name_list = []
    fileno_dict = {}

    # For each selected type...
    for kernel in kernels:

        # Add the name to the end of the list, avoiding duplicates
        name = kernel.full_name
        if name in name_list:
            name_list.remove(name)

        name_list.append(name)

        # If the kernel has a file_no, accumulate a list
        if kernel.file_no is None: continue

        if name not in fileno_dict:
            fileno_dict[name] = []

        if kernel.file_no not in fileno_dict[name]:
            fileno_dict[name].append(kernel.file_no)

    # Attach the file_no ranges to the associated kernel names
    for name in fileno_dict:
        k = name_list.index(name)
        name_list[k] = name + _fileno_str(fileno_dict[name])

    # Return the names
    return name_list

def furnished_names(types=None):
    """Return a list of strings containing the names of the furnished kernels.
    """

    global FURNISHED_NAMES, FURNISHED_FILENOS
    global KERNEL_TYPE_SORT_ORDER

    # Normalize input
    if types is None or types == []:
        types = KERNEL_TYPE_SORT_ORDER
    elif type(types) == str:
        types = [types]

    name_list = []

    # For each selected type...
    for key in types:

        # Walk down list
        for name in FURNISHED_NAMES[key]:
            if name in FURNISHED_FILENOS:
                name_list.append(name + _fileno_str(FURNISHED_FILENOS[name]))
            else:
                name_list.append(name)

    return name_list

def furnished_basenames(types=None):
    """Return a list of strings containing the basenames of the furnished
    kernels."""

    global FURNISHED_NAMES, FURNISHED_FILENOS
    global KERNEL_TYPE_SORT_ORDER

    # Normalize input
    if types is None or types == []:
        types = KERNEL_TYPE_SORT_ORDER
    elif type(types) == str:
        types = [types]

    name_list = []

    # For each selected type...
    for key in types:

        # Walk down list
        for filespec in FURNISHED_ABSPATHS[key]:
            basename = os.path.basename(filespec)
            name_list.append(basename)

    return name_list

def used_basenames(types=[], time=None, bodies=[], sc=None, inst=None,
                             slop=6*60*60):
    """Return a list of SPICE file basenames needed for a particular list of
    bodies and frames at a particular time."""

    global FURNISHED_NAMES, FURNISHED_FILENOS
    global KERNEL_TYPE_SORT_ORDER

    # Normalize input
    if types is None or types == []:
        types = KERNEL_TYPE_SORT_ORDER
    elif type(types) == str:
        types = [types]

    # Normalize time
    if isinstance(time, (str,float,int)):
        time = [time, time]

    time_tai = []
    for tval in time:
        if type(tval) == str:
            time_tai.append(julian.tai_from_iso(tval))
        else:
            time_tai.append(tval)

    # Handle spacecraft and instrument
    ck_needed = False
    if sc:
        bodies.append(sc)
        ck_needed = True

    if inst:
        ck_needed = True
        inst = inst.lower()

    basename_list = []

    # For each selected type...
    for key in types:
      if key == 'CK' and not ck_needed: continue
      if key == 'IK' and not inst: continue

      # Walk down list
      temp_list = []
      for filespec in FURNISHED_ABSPATHS[key]:
        basename = os.path.basename(filespec)
        used = False
        for info in FURNISHED_INFO[basename]:

            if time and info.start_time:
                if time[1] < info.start_tai - slop: continue
                if time[0] > info.stop_tai  + slop: continue

            if bodies and info.spice_id:
                if info.spice_id not in bodies: continue

            used = True

        if used:
            temp_list.append(basename)

      if key == 'IK':
        reduced_list = [name for name in temp_list if inst in name.lower()]
        if reduced_list:
            temp_list = reduced_list

      basename_list += temp_list

    return basename_list

################################################################################
# DEPRECATED: Special kernel loader for Cassini
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

    names = []

    # Leapseconds Kernel (LSK)
    names += furnish_lsk(asof=asof)

    # Instruments and frames
    names += furnish_inst(-82, instrument, asof=asof)

    # Planetary Constants
    bodies = [699] + range(601,654) + [65035, 65040, 65041] # plus a few more
    names += furnish_pck(bodies, asof=asof)

    # Ephemerides (SP Kernels)
    names += furnish_spk(bodies + [-82], time=(start_time,stop_time), asof=asof)

    # C (pointing) Kernels
    names += furnish_ck(-82, time=(start_time, stop_time), asof=asof)

    return names

################################################################################
# Special kernel loader for every planet and moon
################################################################################

def furnish_solar_system(start_time=None, stop_time=None, asof=None,
                         planets=(1,2,3,4,5,6,7,8,9)):
    """A routine designed to load all the SPK, FK and planetary constants files
    needed for the planets and moons of the Solar System.

    Input:
        start_time      the start time of the period of interest, in ISO
                        format, "yyyy-mm-ddThh:mm:ss" or in seconds TAI past
                        January 1, 2000. Use None to furnish the latest kernels
                        irrespective of their time limits.

        stop_time       the stop time of the period of interest.

        asof            an optional earlier date for which values should be
                        returned. Wherever possible, the kernels selected will
                        have release dates earlier than this date. The date is
                        expressed as a string in ISO format.

        planets         1-9 to load kernels for a particular planet and its
                        moons. 0 or None to load nine planets (including Pluto).
                        Use a tuple to list more than one planet number.

    Return:             a list of the names of all the kernels loaded.
    """

    if planets is None or planets == 0:
        planets = (1,2,3,4,5,6,7,8,9)
    if type(planets) == int:
        planets = (planets,)

    names = []

    # Leapseconds Kernel (LSK)
    names += furnish_lsk(asof=asof)

    # Planetary Constants
#     bodies = range(1,11) + range(599, 1000, 100) + [399, 301, 401, 402]
#     bodies += range(501,550) + [55062, 55063]
#     bodies += range(601,654) + [65035, 65040, 65041]    # plus a few more...
#     bodies += range(701,728) + range(801,815) + range(901,906)

    # We speed this up by taking advantage of the fact that certain sets of
    # bodies are always grouped together in the kernels

    bodies = [3, 301, 399]

    if 4 in planets:
        bodies += [4, 401]

    if 5 in planets:
        bodies += [501,505,506,530,540,55062]

    if 6 in planets:
        bodies += [601,610,618,619,633,635,640,65035,65040]

    if 7 in planets:
        bodies += [701,706,715,716,726]

    if 8 in planets:
        bodies += [801,802,803,808,809,813,814]

    if 9 in planets:
        bodies += [901,902,904,905]

    names += furnish_pck(bodies, asof=asof)

    # Ephemerides (SP Kernels)
    names += furnish_spk(bodies, time=(start_time, stop_time), asof=asof)

    return names

################################################################################
################################################################################
# spicedb.py and SPICE.db unit tests
################################################################################
################################################################################

class test_KernelInfo(unittest.TestCase):

  # For reference...
  # ['KERNEL_NAME','KERNEL_VERSION', 'KERNEL_TYPE', 'FILESPEC', 'START_TIME',
  #  'STOP_TIME', 'RELEASE_DATE', 'SPICE_ID', 'LOAD_PRIORITY']

  def runTest(self):

    # Sort based on kernel type
    T0 = '2000-01-01T00:00:00'
    T1 = '2001-01-01T00:00:00'
    T2 = '2002-01-01T00:00:00'
    T3 = '2003-01-01T00:00:00'
    T4 = '2004-01-01T00:00:00'
    T5 = '2005-01-01T00:00:00'
    T6 = '2006-01-01T00:00:00'
    T7 = '2007-01-01T00:00:00'
    T8 = '2008-01-01T00:00:00'
    T9 = '2009-01-01T00:00:00'

    lsk  = KernelInfo(['LSK',  '1', 'LSK',  'file', T0, T1, T2, 0, 1])
    lsk2 = KernelInfo(['LSK',  '1', 'LSK',  'file', T0, T1, T2, 0, 1])
    sclk = KernelInfo(['SCLK', '1', 'SCLK', 'file', T0, T1, T2, 0, 1])
    fk   = KernelInfo(['FK',   '1', 'FK',   'file', T0, T1, T2, 0, 1])
    ik   = KernelInfo(['IK',   '1', 'IK',   'file', T0, T1, T2, 0, 1])
    ck   = KernelInfo(['CK',   '1', 'CK',   'file', T0, T1, T2, 0, 1])
    spk  = KernelInfo(['SPK',  '1', 'SPK',  'file', T0, T1, T2, 0, 1])

    self.assertEqual(lsk, lsk2)
    self.assertTrue(lsk <= lsk2)
    self.assertTrue(lsk >= lsk2)
    self.assertFalse(lsk < lsk2)
    self.assertFalse(lsk > lsk2)
    self.assertFalse(lsk != lsk2)

    self.assertTrue(lsk < sclk)
    self.assertTrue(sclk < ck)
    self.assertTrue(fk < ck)
    self.assertTrue(fk < ik)
    self.assertTrue(ik < spk)

    kernels = [spk, ck, ik, fk, sclk, lsk2, lsk]
    kernels.sort()
    self.assertEqual(kernels, [lsk, lsk2, sclk, fk, ik, spk, ck])

    # Sort based on load priority
    spk1 = KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T2, 0, 1])
    spk2 = KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T2, 0, 2])
    spk3 = KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T2, 0, 3])
    spk4 = KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T2, 0, 4])
    spk5 = KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T2, 0, 5])
    spk6 = KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T2, 0, 6])

    kernels = [spk6, spk5, spk4, spk3, spk2, spk1]
    kernels.sort()
    self.assertEqual(kernels, [spk1, spk2, spk3, spk4, spk5, spk6])

    # Sort including release dates
    lsk1 = KernelInfo(['LSK', '1', 'LSK', 'file', T0, T1, T9, 0, 9])
    spk0 = KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T0, 0, 9])
    spk2 = KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T2, 0, 2])
    spk3 = KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T3, 0, 3])
    spk4 = KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T4, 0, 4])
    spk5 = KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T5, 0, 5])

    # note--spk0 has the highest load priority
    kernels = [spk0, spk5, spk4, spk3, spk2, lsk1]
    kernels.sort()
    self.assertEqual(kernels, [lsk1, spk2, spk3, spk4, spk5, spk0])

    # Sort by name and version
    spk0 = KernelInfo(['AA', '1', 'SPK', 'file', T0, T1, T9, 0, 1])
    spk1 = KernelInfo(['AA', '2', 'SPK', 'file', T0, T1, T9, 0, 1])
    spk2 = KernelInfo(['AA', '3', 'SPK', 'file', T0, T1, T9, 0, 1])
    spk3 = KernelInfo(['BB', '3', 'SPK', 'file', T0, T1, T9, 0, 1])
    spk4 = KernelInfo(['BB', '4', 'SPK', 'file', T0, T1, T9, 0, 1])
    spk5 = KernelInfo(['CC', '4', 'SPK', 'file', T0, T1, T9, 0, 1])

    kernels = [spk5, spk4, spk3, spk2, spk1, spk0]
    kernels.sort()
    self.assertEqual(kernels, [spk0, spk1, spk2, spk3, spk4, spk5])

    # Sort by time ranges
    spk0 = KernelInfo(['SPK', '1', 'SPK', 'file', T4, T7, T9, 0, 1])
    spk1 = KernelInfo(['SPK', '1', 'SPK', 'file', T6, T7, T9, 0, 1])
    spk2 = KernelInfo(['SPK', '1', 'SPK', 'file', T0, T4, T9, 0, 1])
    spk3 = KernelInfo(['SPK', '1', 'SPK', 'file', T1, T4, T9, 0, 1])
    spk4 = KernelInfo(['SPK', '1', 'SPK', 'file', T2, T4, T9, 0, 1])
    spk5 = KernelInfo(['SPK', '1', 'SPK', 'file', T3, T4, T9, 0, 1])

    kernels = [spk0, spk1, spk2, spk3, spk4, spk5]
    kernels.sort()
    self.assertEqual(kernels, [spk5, spk4, spk3, spk2, spk1, spk0])

    # Sort by file name
    spk0 = KernelInfo(['SPK', '1', 'SPK', 'file0', T0, T9, T9, 0, 1])
    spk1 = KernelInfo(['SPK', '1', 'SPK', 'file1', T0, T9, T9, 0, 1])
    spk2 = KernelInfo(['SPK', '1', 'SPK', 'file2', T0, T9, T9, 0, 1])
    spk3 = KernelInfo(['SPK', '1', 'SPK', 'file3', T1, T9, T9, 0, 1])
    spk4 = KernelInfo(['SPK', '1', 'SPK', 'file4', T0, T9, T9, 0, 1])
    spk5 = KernelInfo(['SPK', '1', 'SPK', 'file5', T0, T9, T9, 0, 1])

    kernels = [spk5, spk4, spk3, spk2, spk1, spk0]
    kernels.sort()
    self.assertEqual(kernels, [spk3, spk0, spk1, spk2, spk4, spk5])

    # Sort by body ID
    spk0 = KernelInfo(['SPK', '1', 'SPK', 'file1', T0, T9, T9, 0, 1])
    spk1 = KernelInfo(['SPK', '1', 'SPK', 'file1', T0, T9, T9, 1, 1])
    spk2 = KernelInfo(['SPK', '1', 'SPK', 'file1', T0, T9, T9, 2, 1])
    spk3 = KernelInfo(['SPK', '1', 'SPK', 'file0', T0, T9, T9, 3, 1])
    spk4 = KernelInfo(['SPK', '1', 'SPK', 'file0', T0, T9, T9, 4, 1])
    spk5 = KernelInfo(['SPK', '1', 'SPK', 'file0', T0, T9, T9, 5, 1])

    kernels = [spk5, spk4, spk3, spk2, spk1, spk0]
    kernels.sort()
    self.assertEqual(kernels, [spk3, spk4, spk5, spk0, spk1, spk2])

    # Test full names
    spk = KernelInfo(['VG1-JUP', '+230', 'SPK', 'file', T0, T1, T2, 0, 1])
    self.assertEqual(spk.full_name, 'VG1-JUP230')

    spk = KernelInfo(['VG1', 'JUP230', 'SPK', 'file', T0, T1, T2, 0, 1])
    self.assertEqual(spk.full_name, 'VG1-JUP230')

    spk = KernelInfo(['VG1-JUP230', None, 'SPK', 'file', T0, T1, T2, 0, 1])
    self.assertEqual(spk.full_name, 'VG1-JUP230')

################################################################################
# UNIT TESTS for queries
################################################################################

class test_spicedb(unittest.TestCase):

  def runTest(self):

    global DEBUG, ABSPATH_LIST

    ############################################################################
    # _sort_kernels()
    ############################################################################

    # Leapseconds should always come first
    lsk0 = KernelInfo(['LEAPSECONDS', '1', 'LSK', 'File0.tls',
                       '2000-01-01', '2000-01-02', '2000-01-03', None, 100])

    # Spacecraft clock should always come second
    # These kernels are ordered alphabetically
    sclk0 = KernelInfo(['SCLK82', '1', 'SCLK', 'sclk-82.tsc',
                        '2000-01-01', '2000-01-02', '2003-01-03', -82, 100])

    sclk1 = KernelInfo(['SCLK99', '1', 'SCLK', 'sclk-99.tsc',
                        '2000-01-01', '2000-01-02', '2003-01-03', -99, 100])

    # CKs come next alphabetically
    # Lowest load priority comes first, even with later release date
    ck0 = KernelInfo(['CK-PREDICTED', '1', 'CK', 'File2.ck',
                      '2001-01-01', '2099-01-01', '2005-01-01', -82, 50])

    # Others are loaded in order of increasing end date
    ck1 = KernelInfo(['CK-RECONSTRUCTED', '1', 'CK', 'File3.ck',
                      '2001-01-01', '2002-01-01', '2003-01-01', -82, 100])

    ck2 = KernelInfo(['CK-RECONSTRUCTED', '1', 'CK', 'File4.ck',
                      '2002-01-01', '2003-01-01', '2004-01-01', -82, 100])

    random = [ck2, lsk0, ck1, sclk1, ck0, sclk0]

    sorted = [lsk0, sclk0, sclk1, ck0, ck1, ck2]
    self.assertEqual(_sort_kernels(random), sorted)

    # Frame and PC kernels
    # Ordered by priority, release date, version; with duplicates removed
    fk1 = KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5a.fk',
                      None, None, '2004-01-01', 1, 100])
    fk2 = KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5b.fk',
                      None, None, '2004-01-01', 2, 100])
    fk3 = KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5c.fk',
                      None, None, '2004-01-01', 3, 100])
    # later release date, but only body 1
    fk4 = KernelInfo(['FRAMES', 'BBBB', 'FK', 'File6a.fk',
                      None, None, '2005-01-01', 1, 100])

    random = [fk1, fk2, fk3, fk4]
    sorted = [fk2, fk3, fk4]
    self.assertEqual(_sort_kernels(random), sorted)

    # three bodies in one file
    fk1 = KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5.fk',
                      None, None, '2004-01-01', 1, 100])
    fk2 = KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5.fk',
                      None, None, '2004-01-01', 2, 100])
    fk3 = KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5.fk',
                      None, None, '2004-01-01', 3, 100])

    # later release date, but only body 1
    fk4 = KernelInfo(['FRAMES', 'BBBB', 'FK', 'File6a.fk',
                     None, None, '2005-01-01', 1, 100])

    random = [fk1, fk2, fk3, fk4]
    sorted = [fk3, fk4]
    self.assertEqual(_sort_kernels(random), sorted)

    # higher load priority
    fk1 = KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5.fk',
                      None, None, '2004-01-01', 1, 150])
    fk2 = KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5.fk',
                      None, None, '2004-01-01', 2, 150])
    fk3 = KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5.fk',
                      None, None, '2004-01-01', 3, 150])
    # later release date, but only body 1
    fk4 = KernelInfo(['FRAMES', 'BBBB', 'FK', 'File6a.fk',
                      None, None, '2005-01-01', 1, 100])

    random = [fk1, fk2, fk3, fk4]
    sorted = [fk3]
    self.assertEqual(_sort_kernels(random), sorted)

    # SP Kernels
    # A low-priority predict kernel comes first
    spk1 = KernelInfo(['SPK_PREDICTED', '1', 'SPK', 'predict.spk',
                 '2000-01-02', '2020-12-31', '2003-01-03', -82, 50])

    # These are duplicates and all but the last will be skipped
    spk2 = KernelInfo(['SPK-RECONSTRUCTED', '1', 'SPK', 'recon.spk',
                 '2002-01-01', '2005-01-01', '2003-01-03', -82, 100])

    spk2a = KernelInfo(['SPK-RECONSTRUCTED', '1', 'SPK', 'recon.spk',
                  '2002-01-01', '2005-01-01', '2003-01-03', 6, 100])

    spk2b = KernelInfo(['SPK-RECONSTRUCTED', '1', 'SPK', 'recon.spk',
                  '2002-01-01', '2005-01-01', '2003-01-03', 601, 100])

    spk2c = KernelInfo(['SPK-RECONSTRUCTED', '1', 'SPK', 'recon.spk',
                  '2002-01-01', '2005-01-01', '2003-01-03', 602, 100])

    spk2d = KernelInfo(['SPK-RECONSTRUCTED', '1', 'SPK', 'recon.spk',
                  '2002-01-01', '2005-01-01', '2003-01-03', 699, 100])

    spk2e = KernelInfo(['SPK-RECONSTRUCTED', '1', 'SPK', 'recon.spk',
                  '2002-01-01', '2005-01-01', '2003-01-03', 699, 100])

    # Another SPK, duplicated for three moons, alphabetically earlier
    spk3 = KernelInfo(['SAT123','1',  'SPK', 'sat123.spk',
                 '1950-01-01', '2050-01-02', '2003-01-03', 619, 100])

    spk3a = KernelInfo(['SAT123', '1', 'SPK', 'sat123.spk',
                  '1950-01-01', '2050-01-02', '2003-01-03', 635, 100])

    spk3b = KernelInfo(['SAT123', '1', 'SPK', 'sat123.spk',
                  '1950-01-01', '2050-01-02', '2003-01-03', 636, 100])

    random = [spk1, spk2, spk2a, spk2b, spk2c, spk2d, spk2e, spk3, spk3a, spk3b]
    sorted = [spk1, spk3b, spk2e]
    self.assertEqual(_sort_kernels(random), sorted)

    # Put them all together in a random order
    random = [spk3, ck2, fk3, spk2d, ck0, spk2a, spk2b, fk1, fk2, spk2c, ck1,
              lsk0, spk2e, sclk1, sclk0, spk1, fk4, spk2, spk3b, spk3a]
    sorted = [lsk0, sclk0, sclk1, fk3, spk1, spk3b, spk2e, ck0, ck1, ck2]
    self.assertEqual(_sort_kernels(random), sorted)

    ############################################################################
    # _remove_overlaps()
    ############################################################################

    start_time = '2000-01-01T00:00:00'
    stop_time  = '2010-01-01T00:00:00'

    info0 = KernelInfo(['SPK0', 'V1', 'SPK', '0000.spk',
                  '1950-01-01', '2050-01-01', '2003-01-01', 6, 100])

    info1 = KernelInfo(['SPK1', 'V1', 'SPK', '1111.spk',
                  '1950-01-01', '2002-01-01', '2003-01-02', 6, 100])

    info2 = KernelInfo(['SPK2', 'V1', 'SPK', '2222.spk',
                  '1950-01-01', '2003-01-01', '2003-02-01', 6, 100])

    info3 = KernelInfo(['SPK3', 'V1', 'SPK', '3333.spk',
                  '2002-07-01', '2004-01-01', '2003-03-01', 6, 100])

    info4 = KernelInfo(['SPK4', 'V1', 'SPK', '4444.spk',
                  '1950-01-01', '2002-07-01', '2003-04-01', 6, 100])

    info5 = KernelInfo(['SPK5', 'V1', 'SPK', '5555.spk',
                  '2004-01-01', '2050-01-01', '2003-05-01', 6, 100])

    info6 = KernelInfo(['SPK6', 'V1', 'SPK', '6666.spk',
                  '2004-01-01', '2004-07-01', '2003-06-01', 6, 100])

    info7 = KernelInfo(['SPK7', 'V1', 'SPK', '7777.spk',
                  '1950-01-01', '2050-01-01', '2003-07-01', 6, 100])

    info8 = KernelInfo(['SPK8', 'V1', 'SPK', '8888.spk',
                  '2004-07-01', '2050-01-01', '2003-08-01', 6, 100])

    spks = [info0, info1, info2, info3, info4, info5, info6, info7, info8]
    self.assertEqual(_remove_overlaps(spks, start_time, stop_time),
                                     [info7, info8])

    spks = [info0, info1, info2, info3, info4, info5, info6, info7]
    self.assertEqual(_remove_overlaps(spks, start_time, stop_time),
                     [info7])

    spks = [info0, info1, info2, info3, info4, info5, info6, info8]
    self.assertEqual(_remove_overlaps(spks, start_time, stop_time),
                     [info3, info4, info6, info8])

    spks = [info0, info1, info2, info3, info4, info5, info6]
    self.assertEqual(_remove_overlaps(spks, start_time, stop_time),
                     [info3, info4, info5, info6])

    spks = [info0, info1, info2, info3, info4, info5]
    self.assertEqual(_remove_overlaps(spks, start_time, stop_time),
                     [info3, info4, info5])

    spks = [info0, info1, info2, info3, info4]
    self.assertEqual(_remove_overlaps(spks, start_time, stop_time),
                     [info0, info3, info4])

    spks = [info0, info1, info2, info3]
    self.assertEqual(_remove_overlaps(spks, start_time, stop_time),
                     [info0, info2, info3])

    spks = [info0, info1, info2]
    self.assertEqual(_remove_overlaps(spks, start_time, stop_time),
                     [info0, info2])

    spks = [info0, info1]
    self.assertEqual(_remove_overlaps(spks, start_time, stop_time),
                     [info0, info1])

    spks = [info0]
    self.assertEqual(_remove_overlaps(spks, start_time, stop_time),
                     [info0])

    ############################################################################
    ############################################################################
    # SPICE.db tests
    ############################################################################
    ############################################################################

    try:
        get_spice_path()    # Find SPICE.db in the usual place
        open_db()

        DEBUG = True        # Avoid attempting to load kernels
        ABSPATH_LIST = []

        ########################################################################
        # _query_kernels()
        ########################################################################

        kernels = _query_kernels('LSK')
        self.assertEqual(len(kernels), 1)

        kernels = _query_kernels('LSK', asof='2014-03-09')
        self.assertEqual(kernels[0].full_name, 'NAIF-LSK-0010')

        kernels = _query_kernels('LSK', asof=(14*365.25*86400))
        self.assertEqual(kernels[0].full_name, 'NAIF-LSK-0010')

        kernels = _query_kernels('LSK', asof='2010-01-01')
        self.assertEqual(kernels[0].full_name, 'NAIF-LSK-0009')

        self.assertRaises(ValueError, _query_kernels, 'LSK', asof='1950')

        self.assertRaises(ValueError, _query_kernels, 'LSK', after='3000')

        kernels = _query_kernels('LSK', asof='1950-01-01', redo=True)
        self.assertEqual(len(kernels), 1)
        self.assertTrue(kernels[0].full_name.startswith('NAIF-LSK-'))

        kernels = _query_kernels('LSK', after='3000-01-01', redo=True)
        self.assertEqual(len(kernels), 1)
        self.assertTrue(kernels[0].full_name.startswith('NAIF-LSK-'))

        kernels = _query_kernels('PCK', name='NAIF%')
        self.assertEqual(len(kernels), 1)
        self.assertTrue(kernels[0].full_name.startswith('NAIF-PCK-'))

        kernels = _query_kernels('PCK', body=2)
        self.assertEqual(len(kernels), 1)           # Only NAIF PCKs have Venus
        self.assertTrue(kernels[0].full_name.startswith('NAIF-PCK-'))

        kernels = _query_kernels('PCK', body=2, asof='2014')
        self.assertEqual(kernels[0].full_name, 'NAIF-PCK-00010')

        kernels = _query_kernels('PCK', body=(1,2,3), asof='2014')
        self.assertEqual(kernels[0].full_name, 'NAIF-PCK-00010')

        # Cassini CK tests
        kernels = _query_kernels('CK', body=-82, asof='2014',
                                        time=('2008-01-01','2008-02-01'),
                                        limit=False)
        for kernel in kernels[:-2]:
            self.assertEqual(kernel.full_name, 'CAS-CK-RECONSTRUCTED-V01')

        for kernel in kernels[-2:]:
            self.assertEqual(kernel.full_name, 'CAS-CK-PREDICTED-V01')

        self.assertEqual(len(kernels), 9)
        self.assertTrue(kernels[ 0].filespec.endswith('07362_08002ra.bc'))
        self.assertTrue(kernels[-1].filespec.endswith('08022_08047pg_live.bc'))

        # Cassini SPK tests
        kernels = _query_kernels('SPK', body=-82, asof='2014',
                                        time=('2008-01-01','2009-01-01'),
                                        limit=False)
        for kernel in kernels:
            self.assertEqual(kernel.full_name, 'CAS-SPK-RECONSTRUCTED-V01')

        self.assertEqual(len(kernels), 13)
        self.assertTrue(kernels[ 0].filespec.endswith(
                                        '080327R_SCPSE_07365_08045.bsp'))
        self.assertTrue(kernels[-1].filespec.endswith(
                                        '090225R_SCPSE_08350_09028.bsp'))

        ########################################################################
        # furnish_lsk(asof=None, after=None, redo=True)
        ########################################################################

        ABSPATH_LIST = []
        kernels = furnish_lsk(asof='2014')
        self.assertEqual(kernels, ['NAIF-LSK-0010'])
        self.assertEqual(len(ABSPATH_LIST), 1)
        self.assertTrue(ABSPATH_LIST[0].endswith('/naif0010.tls'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        k1 = kernels
        kernels = furnish_lsk(asof=(14*365.25*86400))
        self.assertEqual(kernels, ['NAIF-LSK-0010'])
        self.assertTrue(ABSPATH_LIST[0].endswith('/naif0010.tls'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_lsk(asof='2010-01-01')
        self.assertEqual(kernels, ['NAIF-LSK-0009'])
        self.assertTrue(ABSPATH_LIST[0].endswith('/naif0009.tls'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        self.assertRaises(ValueError, furnish_lsk, asof='1950', redo=False)

        self.assertRaises(ValueError, furnish_lsk, after='3000', redo=False)

        latest = furnish_lsk()

        kernels = furnish_lsk(asof='1950-01-01', redo=True)
        self.assertEqual(kernels, latest)

        kernels = furnish_lsk(after='3000-01-01', redo=True)
        self.assertEqual(kernels, latest)

        ########################################################################
        # furnish_pck(bodies, asof=None, after=None, redo=True)
        ########################################################################

        ABSPATH_LIST = []
        kernels = furnish_pck()
        naif_pck_mars = -1
        naif_pck = -1
        for (i,kernel) in enumerate(kernels):
            if kernel.startswith('NAIF-PCK-MARS-'): naif_pck_mars = i
            if kernel.startswith('NAIF-PCK-00'): naif_pck = i

        self.assertTrue(naif_pck >= 0)
        self.assertTrue(naif_pck_mars >= 0)

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_pck(bodies=range(1,11) + range(199,1000,100) + [301] +
                                     range(401,403) + range(501,517) +
                                     range(601,654) + range(701,716) +
                                     range(801,808) + range(901,906) +
                                     [814,65035,65040,65041,65045,65048,65050],
                              asof='2014-03-10')

        self.assertEqual(kernels, ['NAIF-PCK-MARS-IAU2000-V0',
                                   'CAS-FK-ROCKS-V18',
                                   'CAS-PCK-ROCKS-2011-01-21',
                                   'CAS-PCK-2014-02-19',
                                   'NAIF-PCK-00010-EDIT-V01'])

        self.assertTrue(ABSPATH_LIST[0].endswith('mars_iau2000_v0.tpc'))
        self.assertTrue(ABSPATH_LIST[1].endswith('cas_rocks_v18.tf'))
        self.assertTrue(ABSPATH_LIST[2].endswith('cpck_rock_21Jan2011_merged.tpc'))
        self.assertTrue(ABSPATH_LIST[3].endswith('cpck19Feb2014.tpc'))
        self.assertTrue(ABSPATH_LIST[4].endswith('pck00010_edit_v01.tpc'))

        ########################################################################
        # furnish_spk(bodies, time=None, asof=None, after=None, redo=True)
        ########################################################################

        ABSPATH_LIST = []
        kernels = furnish_spk([1,2,3,4,5,6,7,8,9], asof='2014-03-10')
        self.assertEqual(kernels, ['DE430'])
        self.assertEqual(len(ABSPATH_LIST), 1)
        self.assertTrue(ABSPATH_LIST[0].endswith('/de430.bsp'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_spk([699], asof='2014-03-10')
        self.assertEqual(kernels, ['SAT363', 'DE430']) #####
        self.assertEqual(len(ABSPATH_LIST), 2)
        self.assertTrue(ABSPATH_LIST[0].endswith('/sat363.bsp'))
        self.assertTrue(ABSPATH_LIST[1].endswith('/de430.bsp'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_spk(range(601,654), asof='2014-03-10')
        self.assertEqual(kernels, ['SAT357', 'SAT360', 'SAT362', 'SAT363',
                                   'DE430'])

        # Only SAT357-rocks is loaded the first time, not SAT357
        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)

        for file in F1:
            self.assertIn(file, ABSPATH_LIST)

        ########
        self.assertRaises(ValueError, furnish_spk, [601], after='3000-01-01',
                                                          redo=False)

        ########
        ABSPATH_LIST = []
        kernels = furnish_spk([601], after='3000-01-01', redo=True)
        self.assertTrue(kernels[0][:3] == 'SAT')
        self.assertTrue(kernels[0][3:] >= '360')
        self.assertTrue(kernels[1][:2] == 'DE')
        self.assertTrue(kernels[1][2:] >= '430')

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_spk([-82], asof='2013-12-13', time=None)
        self.assertEqual(kernels, ['CAS-SPK-RECONSTRUCTED-V01[1-134]'])
        self.assertEqual(len(ABSPATH_LIST), 134)
        self.assertTrue(ABSPATH_LIST[ 0].endswith('/000331R_SK_LP0_V1P32.bsp'))
        self.assertTrue(ABSPATH_LIST[-1].endswith('/131212R_SCPSE_13273_13314.bsp'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_spk([-82], asof='2014-03-10',
                                     time=(12*365.25*86400., '2012-04-01'))
        self.assertEqual(kernels, ['CAS-SPK-RECONSTRUCTED-V01[114-117]'])
        self.assertEqual(len(ABSPATH_LIST), 4)
        self.assertTrue(ABSPATH_LIST[0].endswith('/120227R_SCPSE_11357_12016.bsp'))
        self.assertTrue(ABSPATH_LIST[1].endswith('/120312R_SCPSE_12016_12042.bsp'))
        self.assertTrue(ABSPATH_LIST[2].endswith('/120416R_SCPSE_12042_12077.bsp'))
        self.assertTrue(ABSPATH_LIST[3].endswith('/120426R_SCPSE_12077_12098.bsp'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1, time=(12*365.25*86400., '2012-04-01'))
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_spk([-82], asof='2012-04-01',
                                     time=(12*365.25*86400., '2012-04-01'))
        self.assertEqual(kernels, ['CAS-SPK-PREDICTED-2011-08-18',
                                   'CAS-SPK-RECONSTRUCTED-V01[114,115]'])
        self.assertEqual(len(ABSPATH_LIST), 3)
        self.assertTrue(ABSPATH_LIST[0].endswith('110818AP_SCPSE_11175_17265.bsp'))
        self.assertTrue(ABSPATH_LIST[1].endswith('120227R_SCPSE_11357_12016.bsp'))
        self.assertTrue(ABSPATH_LIST[2].endswith('120312R_SCPSE_12016_12042.bsp'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1, time=(12*365.25*86400., '2012-04-01'))
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_spk([-82], asof='2011-09-01',
                                     time=(12*365.25*86400., '2012-04-01'))
        self.assertEqual(kernels, ['CAS-SPK-PREDICTED-2011-08-18'])
        self.assertEqual(len(ABSPATH_LIST), 1)
        self.assertTrue(ABSPATH_LIST[0].endswith('110818AP_SCPSE_11175_17265.bsp'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1, time=(12*365.25*86400., '2012-04-01'))
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_spk([-82], asof='2011-08-01',
                                     time=(12*365.25*86400., '2012-04-01'))
        self.assertEqual(kernels, ['CAS-SPK-PREDICTED-2009-10-05'])
        self.assertEqual(len(ABSPATH_LIST), 1)
        self.assertTrue(ABSPATH_LIST[0].endswith('091005AP_SCPSE_09248_17265.bsp'))

        self.assertRaises(ValueError, furnish_spk, [-82], after='3000-01-01',
                                                          redo=False)

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1, time=(12*365.25*86400., '2012-04-01'))
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_spk([-32,601,699], asof='2014-08-01',
                              time=('1981-08-14', '1981-08-24'))
        self.assertEqual(kernels, ['SAT360', 'SAT363', 'VG2-SPK-SAT337',
                                   'DE432'])
        self.assertEqual(len(ABSPATH_LIST), 5)
        self.assertTrue(ABSPATH_LIST[-3].endswith('/sat337.bsp'))
        self.assertTrue(ABSPATH_LIST[-2].endswith('/vgr2_sat337.bsp'))
        self.assertTrue('de432' in ABSPATH_LIST[-1])

        self.assertRaises(ValueError, furnish_spk, [-82], after='3000-01-01',
                                                          redo=False)

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1, time=('1981-08-14', '1981-08-24'))
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        sl9 = range(1000181,1000189) + [1000190,1000191] + \
              range(1000193,1000204)
        kernels = furnish_spk(sl9, asof='2014-08-01')
        self.assertEqual(kernels, ['SL9-SPK-DE403'])
        self.assertEqual(len(ABSPATH_LIST), len(sl9) + 1)
        self.assertTrue(ABSPATH_LIST[-1].endswith('/de403.bsp'))
        for file in ABSPATH_LIST[:-1]:
            self.assertTrue(file.endswith('_1992-1994.gst.DE403.bsp'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########################################################################
        # furnish_inst(ids, inst=None, asof=None, after=None, redo=True)
        ########################################################################

        ABSPATH_LIST = []
        kernels = furnish_inst(-82, inst=[], asof='2014-03-10')
        self.assertEqual(kernels, ['CAS-SCLK-00158', 'CAS-FK-V04'])
        self.assertEqual(len(ABSPATH_LIST), 3)
        self.assertTrue(ABSPATH_LIST[0].endswith('/cas00158.tsc'))
        self.assertTrue(ABSPATH_LIST[1].endswith('/cas_v40.tf'))
        self.assertTrue(ABSPATH_LIST[2].endswith('/cas_status_v04.tf'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_inst(-82, inst='ISS', asof='2014-03-10')
        self.assertEqual(kernels, ['CAS-SCLK-00158', 'CAS-FK-V04',
                                   'CAS-IK-ISS-V10'])
        self.assertEqual(len(ABSPATH_LIST), 4)
        self.assertTrue(ABSPATH_LIST[0].endswith('/cas00158.tsc'))
        self.assertTrue(ABSPATH_LIST[1].endswith('/cas_v40.tf'))
        self.assertTrue(ABSPATH_LIST[2].endswith('/cas_status_v04.tf'))
        self.assertTrue(ABSPATH_LIST[3].endswith('/cas_iss_v10.ti'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_inst(-82, inst=None, asof='2014-03-10')
        for file in ABSPATH_LIST[3:]:      # skip over one .tsc and two .tf files
            self.assertTrue(file.endswith('.ti'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_inst(-31, inst='ISS', asof='2014-03-10')
        self.assertEqual(kernels, ['VG1-SCLK-00019', 'VG1-FK-V02',
                                   'VG1-IK-ISSNA-V02', 'VG1-IK-ISSWA-V01'])

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########################################################################
        # furnish_ck(ids, time=None, asof=None, after=None, redo=True)
        ########################################################################

        ABSPATH_LIST = []
        kernels = furnish_ck(-82)
        self.assertEqual(kernels[0], 'CAS-CK-PREDICTED-V01[1-104]')
        self.assertTrue(kernels[1].startswith('CAS-CK-JUP-V01[1-'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_ck(-82, asof='2014-02-18')
        self.assertEqual(kernels, ['CAS-CK-PREDICTED-V01[1-81]',
                                   'CAS-CK-JUP-V01[1-64]',
                                   'CAS-CK-RECONSTRUCTED-V01[1-744]'])
        self.assertEqual(len(ABSPATH_LIST), 81+64+744)

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_ck(-82, time=('2005-01-01','2005-02-01'),
                                  asof='2014-01-01')
        self.assertEqual(kernels, ['CAS-CK-PREDICTED-V01[10,11]',
                                   'CAS-CK-RECONSTRUCTED-V01[69-75]'])
        self.assertEqual(len(ABSPATH_LIST), 2+7)
        self.assertTrue(ABSPATH_LIST[2].endswith('/05002_05007ra.bc'))
        self.assertTrue(ABSPATH_LIST[3].endswith('/05007_05012ra.bc'))
        self.assertTrue(ABSPATH_LIST[4].endswith('/05012_05017ra.bc'))
        self.assertTrue(ABSPATH_LIST[5].endswith('/05017_05022ra.bc'))
        self.assertTrue(ABSPATH_LIST[6].endswith('/05022_05027ra.bc'))
        self.assertTrue(ABSPATH_LIST[7].endswith('/04361_05002ra.bc'))
        self.assertTrue(ABSPATH_LIST[8].endswith('/05027_05032ra.bc'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1, time=('2005-01-01','2005-02-01'))
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_ck(-82, time=('2005-01-01','2005-02-01'),
                                  asof='2005-02-01')
        self.assertEqual(kernels, ['CAS-CK-RECONSTRUCTED-V01[69-73]'])
        self.assertEqual(len(ABSPATH_LIST), 5)
        self.assertTrue(ABSPATH_LIST[0].endswith('/05002_05007ra.bc'))
        self.assertTrue(ABSPATH_LIST[1].endswith('/05007_05012ra.bc'))
        self.assertTrue(ABSPATH_LIST[2].endswith('/05012_05017ra.bc'))
        self.assertTrue(ABSPATH_LIST[3].endswith('/05017_05022ra.bc'))
        self.assertTrue(ABSPATH_LIST[4].endswith('/05022_05027ra.bc'))

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1, time=('2005-01-01','2005-02-01'))
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_ck(-32, asof='2014-01-01')
        self.assertIn('VG2-CK-ISS-JUP-V01', kernels)
        self.assertIn('VG2-CK-ISS-SAT-V01', kernels)
        self.assertIn('VG2-CK-ISS-URA-V01', kernels)
        self.assertIn('VG2-CK-ISS-NEP-V01', kernels)

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########
        ABSPATH_LIST = []
        kernels = furnish_ck(-98, asof='2015-06-01')
        self.assertEqual(kernels, ['NH-CK-RECONSTRUCTED-V01[1-9]'])

        F1 = ABSPATH_LIST
        k1 = kernels
        ABSPATH_LIST = []
        kernels = furnish_by_name(k1)
        self.assertEqual(k1, kernels)
        self.assertEqual(F1, ABSPATH_LIST)

        ########################################################################
        # DEBUG mode off...
        ########################################################################

        DEBUG = False

        ########################################################################
        # furnish_solar_system(start_time, stop_time, asof=None)
        # unload_by_name(names)
        # unload_by_type(types=None)
        ########################################################################

        kernels = furnish_solar_system('2000-01-01', '2020-01-01',
                                       asof='2014-03-10')

        self.assertIn('NAIF-LSK-0010', kernels[0:1])
        self.assertIn('NAIF-PCK-MARS-IAU2000-V0', kernels[1:6])
        self.assertIn('NAIF-PCK-00010-EDIT-V01', kernels[1:6])
        self.assertIn('CAS-FK-ROCKS-V18', kernels[1:6])
        self.assertIn('CAS-PCK-ROCKS-2011-01-21', kernels[1:6])
        self.assertIn('CAS-PCK-2014-02-19', kernels[1:6])
        self.assertIn('MAR097', kernels[6:-1])
        self.assertIn('JUP300', kernels[6:-1])
        self.assertIn('JUP310', kernels[6:-1])
        self.assertIn('SAT357', kernels[6:-1])
        self.assertIn('SAT360', kernels[6:-1])
        self.assertIn('SAT362', kernels[6:-1])
        self.assertIn('SAT363', kernels[6:-1])
        self.assertIn('URA091', kernels[6:-1])
        self.assertIn('URA111', kernels[6:-1])
        self.assertIn('URA112', kernels[6:-1])
        self.assertIn('NEP077', kernels[6:-1])
        self.assertIn('NEP081', kernels[6:-1])
        self.assertIn('NEP086', kernels[6:-1])
        self.assertIn('NEP087', kernels[6:-1])
        self.assertIn('PLU043', kernels[6:-1])
        self.assertIn('DE430', kernels[-1:])

        self.assertEqual(len(kernels), 22)

        self.assertTrue(kernels.index('JUP300') < kernels.index('JUP310'))
        self.assertTrue(kernels.index('SAT357') < kernels.index('SAT360'))
        self.assertTrue(kernels.index('SAT360') < kernels.index('SAT362'))
        self.assertTrue(kernels.index('SAT362') < kernels.index('SAT363'))
        self.assertTrue(kernels.index('URA091') < kernels.index('URA111'))
        self.assertTrue(kernels.index('URA111') < kernels.index('URA112'))
        self.assertTrue(kernels.index('NEP077') < kernels.index('NEP081'))
        self.assertTrue(kernels.index('NEP081') < kernels.index('NEP087'))
        self.assertTrue(kernels.index('NEP087') < kernels.index('NEP086'))
        # NEP087 < NEP086 because the latter has the later creation date

        self.assertEqual(len(FURNISHED_ABSPATHS['LSK']), 1)
        self.assertEqual(len(FURNISHED_ABSPATHS['PCK']), 5)
        self.assertEqual(len(FURNISHED_ABSPATHS['SPK']), 16)
        self.assertEqual(len(FURNISHED_NAMES['LSK']), 1)
        self.assertEqual(len(FURNISHED_NAMES['PCK']), 5)
        self.assertEqual(len(FURNISHED_NAMES['SPK']), 16)

        unload_by_name(kernels[:6])
        self.assertEqual(len(FURNISHED_ABSPATHS['LSK']), 0)
        self.assertEqual(len(FURNISHED_ABSPATHS['PCK']), 0)
        self.assertEqual(len(FURNISHED_ABSPATHS['SPK']), 16)
        self.assertEqual(len(FURNISHED_NAMES['LSK']), 0)
        self.assertEqual(len(FURNISHED_NAMES['PCK']), 0)
        self.assertEqual(len(FURNISHED_NAMES['SPK']), 16)

        unload_by_type('SPK')
        self.assertEqual(len(FURNISHED_ABSPATHS['SPK']), 0)
        self.assertEqual(len(FURNISHED_NAMES['SPK']), 0)

        ########
        kernels1 = furnish_solar_system(asof='2014-03-10')
        self.assertEqual(kernels, kernels1)

        ########################################################################
        # furnish_cassini_kernels(start_time, stop_time, instrument=None,
        #                         asof=None)
        # unload_by_name(names)
        # unload_by_type(types=None)
        # furnished_names(types=None)
        ########################################################################

        unload_by_type()
        kernels = furnish_cassini_kernels('2010-01-01', '2010-04-01',
                                          instrument='ISS', asof='2014-03-10')

        self.assertIn('NAIF-LSK-0010', kernels[0:1])
        self.assertIn('CAS-SCLK-00158', kernels[1:2])
        self.assertIn('CAS-FK-V04', kernels[2:4])
        self.assertIn('CAS-IK-ISS-V10', kernels[2:4])
        self.assertIn('CAS-FK-ROCKS-V18', kernels[4:8])
        self.assertIn('CAS-PCK-ROCKS-2011-01-21', kernels[4:8])
        self.assertIn('CAS-PCK-2014-02-19', kernels[4:8])
        self.assertIn('NAIF-PCK-00010-EDIT-V01', kernels[4:8])
        self.assertIn('SAT357', kernels[8:12])
        self.assertIn('SAT360', kernels[8:12])
        self.assertIn('SAT362', kernels[8:12])
        self.assertIn('SAT363', kernels[8:12])
        self.assertIn('CAS-SPK-RECONSTRUCTED-V01[90-94]', kernels[12:13])
        self.assertIn('DE430', kernels[13:14])
        self.assertIn('CAS-CK-RECONSTRUCTED-V01[438-456]', kernels[-1:])

        self.assertEqual(FURNISHED_NAMES['LSK'], ['NAIF-LSK-0010'])
        self.assertEqual(FURNISHED_NAMES['SCLK'], ['CAS-SCLK-00158'])
        self.assertEqual(FURNISHED_NAMES['FK'], ['CAS-FK-V04'])
        self.assertEqual(FURNISHED_NAMES['IK'], ['CAS-IK-ISS-V10'])

        self.assertIn('CAS-FK-ROCKS-V18', FURNISHED_NAMES['PCK'])
        self.assertIn('CAS-PCK-ROCKS-2011-01-21', FURNISHED_NAMES['PCK'])
        self.assertIn('CAS-PCK-2014-02-19', FURNISHED_NAMES['PCK'])
        self.assertIn('NAIF-PCK-00010-EDIT-V01', FURNISHED_NAMES['PCK'])
        self.assertEqual(len(FURNISHED_ABSPATHS['PCK']), 4)

        self.assertIn('SAT357', FURNISHED_NAMES['SPK'][:4])
        self.assertIn('SAT360', FURNISHED_NAMES['SPK'][:4])
        self.assertIn('SAT362', FURNISHED_NAMES['SPK'][:4])
        self.assertIn('SAT363', FURNISHED_NAMES['SPK'][:4])
        self.assertIn('CAS-SPK-RECONSTRUCTED-V01', FURNISHED_NAMES['SPK'][4:5])
        self.assertIn('DE430', FURNISHED_NAMES['SPK'][5:6])
        self.assertEqual(FURNISHED_FILENOS['CAS-SPK-RECONSTRUCTED-V01'],
                         range(90,95))
        self.assertEqual(len(FURNISHED_ABSPATHS['SPK']), 5 + 5)

        self.assertEqual(FURNISHED_NAMES['CK'], ['CAS-CK-PREDICTED-V01',
                                                 'CAS-CK-RECONSTRUCTED-V01'])
        self.assertEqual(FURNISHED_FILENOS['CAS-CK-PREDICTED-V01'],
                         range(59,62))
        self.assertEqual(FURNISHED_FILENOS['CAS-CK-RECONSTRUCTED-V01'],
                         range(438,457))
        self.assertEqual(len(FURNISHED_ABSPATHS['CK']), 457 - 438 + 62 - 59)

        ########
        kernels1 = furnish_cassini_kernels('2010-03-01', '2010-06-01',
                                          instrument='VIMS', asof='2014-03-10')

        self.assertIn('NAIF-LSK-0010', kernels1[0:1])
        self.assertIn('CAS-SCLK-00158', kernels1[1:2])
        self.assertIn('CAS-FK-V04', kernels1[2:4])
        self.assertIn('CAS-IK-VIMS-V06', kernels1[2:4])
        self.assertIn('CAS-FK-ROCKS-V18', kernels1[4:8])
        self.assertIn('CAS-PCK-ROCKS-2011-01-21', kernels1[4:8])
        self.assertIn('CAS-PCK-2014-02-19', kernels1[4:8])
        self.assertIn('NAIF-PCK-00010-EDIT-V01', kernels1[4:8])
        self.assertIn('SAT357', kernels1[8:12])
        self.assertIn('SAT360', kernels1[8:12])
        self.assertIn('SAT362', kernels1[8:12])
        self.assertIn('SAT363', kernels1[8:12])
        self.assertIn('CAS-SPK-RECONSTRUCTED-V01[93-97]', kernels1[12:13])
        self.assertIn('DE430', kernels1[13:14])
        self.assertIn('CAS-CK-RECONSTRUCTED-V01[450-468]', kernels1[-1:])

        self.assertEqual(FURNISHED_NAMES['LSK'], ['NAIF-LSK-0010'])
        self.assertEqual(FURNISHED_NAMES['SCLK'], ['CAS-SCLK-00158'])
        self.assertEqual(FURNISHED_NAMES['FK'], ['CAS-FK-V04'])
        self.assertEqual(FURNISHED_NAMES['IK'], ['CAS-IK-ISS-V10',
                                                 'CAS-IK-VIMS-V06'])

        self.assertIn('CAS-FK-ROCKS-V18', FURNISHED_NAMES['PCK'])
        self.assertIn('CAS-PCK-ROCKS-2011-01-21', FURNISHED_NAMES['PCK'])
        self.assertIn('CAS-PCK-2014-02-19', FURNISHED_NAMES['PCK'])
        self.assertIn('NAIF-PCK-00010-EDIT-V01', FURNISHED_NAMES['PCK'])
        self.assertEqual(len(FURNISHED_ABSPATHS['PCK']), 4)

        self.assertIn('SAT357', FURNISHED_NAMES['SPK'][:4])
        self.assertIn('SAT360', FURNISHED_NAMES['SPK'][:4])
        self.assertIn('SAT362', FURNISHED_NAMES['SPK'][:4])
        self.assertIn('SAT363', FURNISHED_NAMES['SPK'][:4])
        self.assertIn('CAS-SPK-RECONSTRUCTED-V01', FURNISHED_NAMES['SPK'][4:5])
        self.assertIn('DE430', FURNISHED_NAMES['SPK'][5:6])
        self.assertEqual(FURNISHED_FILENOS['CAS-SPK-RECONSTRUCTED-V01'],
                         range(90,98))
        self.assertEqual(len(FURNISHED_ABSPATHS['SPK']), 8 + 5)

        self.assertEqual(FURNISHED_NAMES['CK'], ['CAS-CK-PREDICTED-V01',
                                                 'CAS-CK-RECONSTRUCTED-V01'])
        self.assertEqual(FURNISHED_FILENOS['CAS-CK-PREDICTED-V01'],
                         range(59,64))
        self.assertEqual(FURNISHED_FILENOS['CAS-CK-RECONSTRUCTED-V01'],
                         range(438,469))
        self.assertEqual(len(FURNISHED_ABSPATHS['CK']), 469 - 438 + 64 - 59)
        # SPK and CK file_no lists get merged

        ########
        self.assertEqual(furnished_names('CK'),
                         ['CAS-CK-PREDICTED-V01[59-63]',
                          'CAS-CK-RECONSTRUCTED-V01[438-468]'])

        unload_by_name('CAS-CK-RECONSTRUCTED-V01[440]')

        self.assertEqual(furnished_names('CK'),
                         ['CAS-CK-PREDICTED-V01[59-63]',
                          'CAS-CK-RECONSTRUCTED-V01[438,439,441-468]'])
        self.assertEqual(FURNISHED_FILENOS['CAS-CK-RECONSTRUCTED-V01'],
                         [438,439] + range(441,469))

        unload_by_name('CAS-CK-RECONSTRUCTED-V01[1-438]')

        self.assertEqual(furnished_names('CK'),
                         ['CAS-CK-PREDICTED-V01[59-63]',
                          'CAS-CK-RECONSTRUCTED-V01[439,441-468]'])
        self.assertEqual(FURNISHED_FILENOS['CAS-CK-RECONSTRUCTED-V01'],
                         [439] + range(441,469))

        unload_by_name('CAS-CK-RECONSTRUCTED-V01[439,442-465]')

        self.assertEqual(furnished_names('CK'),
                         ['CAS-CK-PREDICTED-V01[59-63]',
                          'CAS-CK-RECONSTRUCTED-V01[441,466-468]'])
        self.assertEqual(FURNISHED_FILENOS['CAS-CK-RECONSTRUCTED-V01'],
                         [441] + range(466,469))

        unload_by_name('CAS-CK-RECONSTRUCTED-V01[441-468]')

        self.assertEqual(furnished_names('CK'), ['CAS-CK-PREDICTED-V01[59-63]'])
        self.assertNotIn('CAS-CK-RECONSTRUCTED-V01', FURNISHED_FILENOS)

        self.assertEqual(furnished_names('SPK'),
                         ['SAT357', 'SAT360', 'SAT362', 'SAT363',
                          'CAS-SPK-RECONSTRUCTED-V01[90-97]', 'DE430'])

        self.assertEqual(furnished_names(['IK','FK','LSK','SCLK']),
                         ['CAS-IK-ISS-V10', 'CAS-IK-VIMS-V06',
                          'CAS-FK-V04', 'NAIF-LSK-0010', 'CAS-SCLK-00158'])

        ########################################################################
        # Test translator
        ########################################################################

        unload_by_type()

        DEBUG = True

        # Function to translate Cassini SPKs, adding "_testing" before suffix
        # and replacing the leading directory path with 'my_testing/'
        def translator(filepath):
          if filepath.endswith('.bsp') and 'RECONSTRUCTED' in filepath.upper():
            lpref = len(get_spice_path())
            return 'my_testing/' + filepath[lpref:-4] + '_testing.bsp'
          return filepath

        # Translator will not affect solar system kernels
        ABSPATH_LIST = []
        kernels1 = furnish_solar_system('2000-01-01', '2020-01-01',
                                        asof='2014-03-10')
        abspaths1 = set(ABSPATH_LIST)
        unload_by_type()

        set_translator(translator)
        ABSPATH_LIST = []
        kernels2 = furnish_solar_system('2000-01-01', '2020-01-01',
                                        asof='2014-03-10')
        abspaths2 = set(ABSPATH_LIST)
        unload_by_type()

        self.assertEqual(kernels1, kernels2)
        self.assertEqual(abspaths1, abspaths2)

        # Translator will change Cassini SPKs
        set_translator(None)
        ABSPATH_LIST = []
        kernels1 = furnish_cassini_kernels('2010-01-01', '2010-04-01',
                                           instrument='ISS', asof='2014-03-10')
        abspaths1 = set(ABSPATH_LIST)
        unload_by_type()

        set_translator(translator)
        ABSPATH_LIST = []
        kernels2 = furnish_cassini_kernels('2010-01-01', '2010-04-01',
                                           instrument='ISS', asof='2014-03-10')
        abspaths2 = set(ABSPATH_LIST)
        unload_by_type()

        self.assertEqual(kernels1, kernels2)

        translated = abspaths2 - abspaths1
        originals  = abspaths1 - abspaths2

        remainder1 = abspaths1 - originals
        remainder2 = abspaths2 - translated
        self.assertEqual(remainder1, remainder2)

        for abspath in originals:
            self.assertTrue(abspath.endswith('.bsp'))
            self.assertTrue('CASSINI' in abspath.upper())

            newpath = translator(abspath)
            self.assertTrue(newpath in translated)

        self.assertTrue(len(translated) == len(originals))

        # Function to replace all files "*.bc" with a blank string
        def translator2(filepath):
            if filepath.endswith('.bc') :
                return ''
            return filepath

        # Translator will eliminate all C kernels from list
        set_translator(translator2)
        ABSPATH_LIST = []
        kernels2 = furnish_cassini_kernels('2010-01-01', '2010-04-01',
                                           instrument='ISS', asof='2014-03-10')
        abspaths2 = set(ABSPATH_LIST)
        unload_by_type()

        for abspath in abspaths2:
            if abspath.endswith('.bc'):
                self.assertNotIn(abspath, abspaths1)
            else:
                self.assertIn(abspath, abspaths1)

    ############################################################################
    # Clean up...
    ############################################################################

    finally:
        unload_by_type()

        DEBUG = False
        close_db()

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

