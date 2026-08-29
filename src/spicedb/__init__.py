##########################################################################################
# spicedb/__init__.py
#
# This set of routines handles the selection of SPICE kernels based on various
# criteria related to body, instrument, time frame, etc. It also sorts selected
# kernels into their proper load order.
##########################################################################################

import datetime
import numbers
import os
from pathlib import Path
import warnings

from filecache import FCPath, FileCache
import interval
import julian
import textkernel

import cspyce

import spicedb.sqlite_db as db

__all__ = ['lrange', 'KernelInfo', 'kernels_from_filespec', 'set_spice_path',
           'get_spice_path', 'get_spice_filecache',
           'get_spice_filecache_prefix', 'open_db', 'close_db', 'db_is_open',
           'set_translator', 'select_lsk', 'select_pck', 'select_spk',
           'select_inst', 'select_ck', 'select_by_name', 'select_by_filespec',
           'as_dict', 'furnish_kernels', 'furnish_lsk', 'furnish_pck',
           'furnish_spk', 'furnish_inst', 'furnish_ck', 'furnish_by_name',
           'furnish_by_metafile', 'furnish_by_filepath', 'unload_by_name',
           'unload_by_type', 'unload_by_filepath', 'unload_all', 'as_names',
           'furnished_names', 'furnished_basenames', 'used_basenames',
           'furnish_cassini_kernels', 'furnish_solar_system', 'test_KernelInfo',
           'test_spicedb']

# For testing and debugging
DEBUG = False   # If true, no files are furnished.
ABSPATH_LIST = []   # If DEBUG, lists the files that would have been furnished.

IS_OPEN = False
DB_PATH = ''

SPICE_FILECACHE_SHARED_NAME = "oops_kernels"
SPICE_FILECACHE = None
SPICE_FILECACHE_PREFIX = None

TRANSLATOR = None   # Optional user-specified function to alter the absolute
                    # paths of SPICE kernels. This can be used to override the
                    # default kernels to be loaded. See set_translator().
TRANSLATOR_ID = None

# Sometimes you really just want a list
def lrange(*args):
    return list(range(*args))

##########################################################################################
# Global variables to track loaded kernels
##########################################################################################

# Furnished kernel names by type, listed in load order
FURNISHED_NAMES = {
    'CK':   [],
    'FK':   [],
    'IK':   [],
    'LSK':  [],
    'PCK':  [],
    'SCLK': [],
    'SPK':  [],
    'STARS':[],
    'META': [],
    'UNK':  [],
}

# Furnished kernel file paths and names by type, listed in load order
# This really does store local paths, not paths relative to a prefix
FURNISHED_ABSPATHS = {
    'CK':   [],
    'FK':   [],
    'IK':   [],
    'LSK':  [],
    'PCK':  [],
    'SCLK': [],
    'SPK':  [],
    'STARS':[],
    'META': [],
    'UNK':  [],
}

# Furnished file numbers by name.
FURNISHED_FILENOS = {}

# Furnished sets of kernel file info objects, keyed by basename
FURNISHED_INFO = {}

SPICE_PATH = None

##########################################################################################
# Kernel Information class
##########################################################################################

TABLE_NAME = "SPICEDB"
COLUMN_NAMES = ["KERNEL_NAME", "KERNEL_VERSION", "KERNEL_TYPE",
                "FILESPEC", "START_TIME", "STOP_TIME", "RELEASE_DATE",
                "SPICE_ID", "LOAD_PRIORITY", "FILE_NO"]

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
    """Class to manage information about individual SPICE kernels as described by a row of
    the SPICEDB table. It has the property that objects sort into an appropriate order for
    furnishing.
    """

    def __init__(self, info):
        """Info is a list or tuple containing the contents of one row of the
        SPICEDB table. The order of items is defined by the COLUMN_NAMES list
        above, which corresponds to the order of the columns in the table.
        """

        self.kernel_name    = info[KERNEL_NAME_INDEX]
        self.kernel_version = info[KERNEL_VERSION_INDEX]
        self.kernel_type    = info[KERNEL_TYPE_INDEX]
        self.filespec       = info[FILESPEC_INDEX]
        self.start_time     = info[START_TIME_INDEX]
        self.stop_time      = info[STOP_TIME_INDEX]
        self.release_date   = info[RELEASE_DATE_INDEX]
        self.spice_id       = info[SPICE_ID_INDEX]
        self.load_priority  = info[LOAD_PRIORITY_INDEX]
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

        if len(info) > FILE_NO_INDEX:
            self.file_no = info[FILE_NO_INDEX]
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

        if self_type < other_type:
            return -1
        if self_type > other_type:
            return +1

        # Other kernel types are organized alphabetically for no particular
        # reason except to keep kernels of the same type together
        if self.kernel_type < other.kernel_type:
            return -1
        if self.kernel_type > other.kernel_type:
            return +1

        # Compare load priorities
        if self.load_priority < other.load_priority:
            return -1
        if self.load_priority > other.load_priority:
            return +1

        # Compare release dates
        if self.release_date is not None and other.release_date is not None:
            if self.release_date < other.release_date:
                return -1
            if self.release_date > other.release_date:
                return +1

        # Group names alphabetically
        if self.kernel_name < other.kernel_name:
            return -1
        if self.kernel_name > other.kernel_name:
            return +1

        # Earlier versions go first
        if self.kernel_version is not None and other.kernel_version is not None:
            if self.kernel_version < other.kernel_version:
                return -1
            if self.kernel_version > other.kernel_version:
                return +1

        # Earlier file numbers go first
        if self.file_no is not None and other.file_no is not None:
            if self.file_no < other.file_no:
                return -1
            if self.file_no > other.file_no:
                return +1

        # Earlier end dates, later starts go first for better chance of override
        if self.stop_time is not None and other.stop_time is not None:
            if self.stop_time < other.stop_time:
                return -1
            if self.stop_time > other.stop_time:
                return +1

        if self.start_time is not None and other.start_time is not None:
            if self.start_time > other.start_time:
                return -1
            if self.start_time < other.start_time:
                return +1

        # Organize by file name if appropriate
        if self.filespec < other.filespec:
            return -1
        if self.filespec > other.filespec:
            return +1

        # Finally, organize by file name and SPICE ID
        if self.spice_id is not None and other.spice_id is not None:
            if self.spice_id < other.spice_id:
                return -1
            if self.spice_id > other.spice_id:
                return +1

        # If all else fails, they're the same
        return 0

    ######################################################################################
    # Comparison operators, needed for sorting, etc. Note __cmp__ is deprecated.
    ######################################################################################

    def __eq__(self, other):
        if type(self) != type(other):
            return False
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

    def __str__(self):
        return self.__repr__()

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
            for (start_tdb, stop_tdb) in coverages.as_intervals():
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

##########################################################################################
# Kernel List Manipulations
##########################################################################################

def _sort_kernels(kernel_list):
    """Sort a list of KernelInfo objects immediately prior to loading.

    Returns:
        (list): In which duplicates are removed and the rest are sorted into their proper
            load order.
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
        if not timeless_by_name[namekey]:
            continue
        if bodies_by_name[namekey] == {None}:
            continue

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
        if namekey not in namekeys:
            continue

        if kernel.spice_id not in bodies_by_name[namekey]:
            continue

        for (k,filtered) in enumerate(filtered_list):
            if filtered.filespec == kernel.filespec:
                del filtered_list[k]
                break

        filtered_list.append(kernel)

    return filtered_list

def _remove_overlaps(kernel_list, start_time, stop_time):
    """Filter out kernels completely overridden by higher-priority kernels.

    For kernels that have time limits such as CKs and SPKs, this method determines which
    kernels overlap higher-priority kernels, and removes kernels from the list if they are
    not required. It returns the filtered list in the proper load order.

    Parameters:
        start_time (str): The start time of the interval of interest, as ISO format
            "yyyy-hh-mmThh:mm:ss" or as seconds TAI since January 1, 2000. None to ignore
            time limits and just select the most recent kernel(s).
        stop_time (str): The stop time of the interval of interest. None to ignore time
            limits.  and just select the most recent kernel(s).

    Returns:
        A filtered list of kernels, in which unnecessary kernels have been removed.
            An unnecessary kernel is one whose entire time range is covered by
            higher-priority kernels.
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
    if isinstance(start_time, str):
        interval_start_tai = julian.tai_from_iso(start_time)
    else:
        interval_start_tai = start_time

    if isinstance(stop_time, str):
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
    split_by_commas = name.index.split(',')
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

##########################################################################################
# Database Query Support
##########################################################################################

def _query_kernels(kernel_type, name=None, body=None, time=None, asof=None,
                                after=None, path=None, limit=True, redo=False):
    """Return a list of KernelInfo objects based on the given constraints.

    Parameters:
        name (str, optional): A SQL match string for the name of the kernel; use "%" for
            multiple wildcards and "_" for a single wildcard.
        body (int, list or tuple, optional): Zero or more SPICE body IDs.
        time (tuple, optional): Consisting of a start and stop time, each expressed as a
            string in ISO format, "yyyy-mm-ddThh:mm:ss". Alternatively, times may be given
            as elapsed seconds TAI since January 1, 2000.
        asof (str, optional): An optional date earlier than today for which values should
            be returned. Wherever possible, the kernels selected will have release dates
            earlier than this date. The date is expressed as a string in ISO format or as
            a number of seconds TAI elapsed since January 1, 2000.
        after (str, optional): An optional date such that files originating earlier are
            not considered. The date is expressed as a string in ISO format or as a number
            of seconds TAI elapsed since January 1, 2000.
        path (str, optional): That must appear within the file specification path of the
            kernel.
        limit (bool, optional): True to limit the number of returned kernels to one where
            appropriate; False to return all the matching kernels.

    Returns:
        (list): A list of KernelInfo objects describing the files that match the
            requirements.
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

    Parameters:
        name (str, optional): A SQL match string for the name of the kernel; use "%" for
            multiple wildcards and "_" for a single wildcard.
        body (int, list or tuple, optional): One or more SPICE body IDs.
        time (tuple, optional): Consisting of a start and stop time, each expressed as a
            string in ISO format "yyyy-mm-ddThh:mm:ss". Alternatively, times may be given
            as elapsed seconds TAI since January 1, 2000.
        asof (str, optional): An optional date earlier than today for which values should
            be returned. Wherever possible, the kernels selected will have release dates
            earlier than this date. The date is expressed as a string in ISO format or as
            a number of seconds TAI elapsed since January 1, 2000.
        after (str, optional): An optional date such that files originating earlier are
            not considered. The date is expressed as a string in ISO format or as a number
            of seconds TAI elapsed since January 1, 2000.
        path (str, optional): That must appear within the file specification path of the
            kernel.
        limit (bool, optional): True to limit the number of returned kernels to one where
            appropriate; False to return all the matching kernels.

    Returns:
        A complete SQL query string.
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
        if isinstance(body, numbers.Integral):
            query_list += ["AND SPICE_ID = ", str(body), "\n"]
            bodies = 1
        else:
            bodies = len(body)

            if bodies == 0:
                pass
            elif bodies == 1:
                query_list += ["AND SPICE_ID = ", str(body[0]), "\n"]
            else:
                query_list += ["AND SPICE_ID in (", str(list(body))[1:-1],
                               ")\n"]

    # Insert start and stop times
    if time is None:
        time = (None, None)

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

    # Insert 'after' constraint except on second pass
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

    Parameters:
        names (str, list or tuple): One or more full kernel names, including versions,
            optionally indexed by file_no ranges.
        time (tuple, optional): Consisting of a start and stop time, each expressed as a
            string in ISO format, "yyyy-mm-ddThh:mm:ss". Alternatively, times may be given
            as elapsed seconds TAI since January 1, 2000. Use None to return kernels
            regardless of the time.

    Returns:
        (list): A list of KernelInfo objects describing the files that match the
            requirements.
    """

    # Normalize the input
    if isinstance(names, str):
        names = [names]

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

    Parameters:
        name (str): A full kernel name including version, optionally indexed by file_no
            ranges.
        time (tuple, optional): Consisting of a start and stop time, each expressed as a
            string in ISO format, "yyyy-mm-ddThh:mm:ss". Alternatively, times may be given
            as elapsed seconds TAI since January 1, 2000. Use None to return kernels
            regardless of the time.

    Returns:
        (list): A list of KernelInfo objects describing the files that match the
            requirements.
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
    if time is None:
        time = (None, None)

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

    Parameters:
        filespec (str or FCPath): One file path or match pattern.
        time (tuple, optional): Consisting of a start and stop time, each expressed as a
            string in ISO format, "yyyy-mm-ddThh:mm:ss". Alternatively, times may be given
            as elapsed seconds TAI since January 1, 2000. Use None to return kernels
            regardless of the time.

    Returns:
        (list): A list of KernelInfo objects describing the files that match the pattern.
    """

    # Normalize the input
    if isinstance(filespecs, str):
        filespecs = [filespecs]

    # Loop through names...
    kernel_info = []

    for filespec in filespecs:

        # Query the database
        sql_string = _sql_query_by_filespec(filespec, time)

        table = db.query(sql_string)

        # If we have nothing, raise an exception
        if len(table) == 0:
            if time is not None:        # Maybe it's just out of time range
                sql_string = _sql_query_by_filespec(filespec)
                table2 = db.query(sql_string)
                if len(table2) > 0:
                    continue

            raise ValueError("no results found matching query")

        for row in table:
            kernel_info.append(KernelInfo(row))

    return kernel_info

def _sql_query_by_filespec(filespec, time=None):
    """Generate a query string based on a kernel name.

    Parameters:
        filespec (str or FCPath): One file path or match pattern.
        time (tuple, optional): Consisting of a start and stop time, each expressed as a
            string in ISO format, "yyyy-mm-ddThh:mm:ss". Alternatively, times may be given
            as elapsed seconds TAI since January 1, 2000. Use None to return kernels
            regardless of the time.

    Returns:
        (list): A list of KernelInfo objects describing the files that match the
            requirements.
    """

    # Begin query
    query_list  = ["SELECT ", COLUMN_STRING, " FROM SPICEDB\n"]
    query_list += ["WHERE FILESPEC like '%", filespec, "'\n"]

    # Insert start and stop times
    if time is None:
        time = (None, None)

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

##########################################################################################
##########################################################################################
# Public API
##########################################################################################
##########################################################################################

def set_spice_path(spice_path=""):
    """Define the directory path to the root of the SPICE file directory tree.

    This directory may also be on a webserver or in the cloud by providing an
    appropriate prefix.

    Call with no argument to reset the path to its default value.
    """

    global SPICE_PATH, SPICE_FILECACHE, SPICE_FILECACHE_PFX

    SPICE_PATH = spice_path
    SPICE_FILECACHE = None
    SPICE_FILECACHE_PFX = None

def get_spice_path():
    """Return the current path to the root of the SPICE file directory tree.

    This directory may also be on a webserver or in the cloud by providing an
    appropriate prefix.

    If the path is undefined, it uses the value of environment variable
    SPICE_PATH. If SPICE_PATH is undefined, it uses ${OOPS_RESOURCES}/SPICE.
    """

    global SPICE_PATH

    if SPICE_PATH is None:
        try:
            SPICE_PATH = FCPath(os.environ["SPICE_PATH"])
        except KeyError:
            SPICE_PATH = FCPath(os.environ["OOPS_RESOURCES"]) / "SPICE"

    return SPICE_PATH

def get_spice_filecache():
    """Return the FileCache used for storing the SPICE DB and kernels."""

    global SPICE_FILECACHE

    if SPICE_FILECACHE is None:
        SPICE_FILECACHE = FileCache(SPICE_FILECACHE_SHARED_NAME)

    return SPICE_FILECACHE

def get_spice_filecache_prefix():
    """Return the FileCachePrefix used for storing the SPICE kernels."""

    global SPICE_FILECACHE_PREFIX

    if SPICE_FILECACHE_PREFIX is None:
        fc = get_spice_filecache()
        spice_path = get_spice_path()
        SPICE_FILECACHE_PREFIX = fc.new_path(spice_path)

    return SPICE_FILECACHE_PREFIX

def open_db(name=None):
    """Open the SPICE database given its name or file path.

    If no name is given, the value of the environment variable
    SPICE_SQLITE_DB_NAME is used. If SPICE_SQLITE_DB_NAME is not set,
    then ${SPICE_PATH}/SPICE.db is used.
    """

    global IS_OPEN, DB_PATH

    if IS_OPEN:
        return

    if name is None:
        if DB_PATH:
            name = DB_PATH
        else:
            try:
                name = FCPath(os.environ["SPICE_SQLITE_DB_NAME"])
            except KeyError:
                name = get_spice_path() / "SPICE.db"

    fc = get_spice_filecache()
    local_path = name.retrieve()  # name will include the URI prefix, if any
    db.open(local_path)
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

##########################################################################################
# Filename translator control
##########################################################################################

def set_translator(func):
    """Define the translator function."""

    global TRANSLATOR, TRANSLATOR_ID

    # Don't worry about a re-definition using the same func
    if TRANSLATOR_ID == id(func):
        return

    if TRANSLATOR and not DEBUG:
        raise RuntimeError('spicedb translator can only be defined once')

    if FURNISHED_INFO and not DEBUG:
        raise RuntimeError('spicedb translator cannot be defined after ' +
                           'kernels have already been loaded.')

    TRANSLATOR = func
    TRANSLATOR_ID = id(func)

##########################################################################################
# Public API for selecting kernels, returning lists of KernelInfo objects
##########################################################################################

def select_lsk(asof=None, after=None, redo=True):
    """Return a sorted list of leapseconds kernels.

    Parameters:
        asof (str, optional): An optional earlier date for which values should be
            returned. Wherever possible, the kernels selected will have release dates
            earlier than this date. The date is expressed as a string in ISO format or as
            a number of seconds TAI elapsed since January 1, 2000.
        after (str, optional): An optional date such that files originating earlier are
            not considered. The date is expressed as a string in ISO format or as a number
            of seconds TAI elapsed since January 1, 2000.
        redo (bool, optional): True to relax the 'asof' and 'after" constraints if no
            matching results are found; False to raise a ValueError instead.

    Returns:
        A sorted list of KernelInfo objects.
    """

    # Search the database
    kernel_list = _query_kernels("LSK", asof=asof, after=after, redo=redo,
                                        limit=True)

    # Load the kernels and return the names
    return _sort_kernels(kernel_list)

def select_pck(bodies=None, name=None, asof=None, after=None, redo=True):
    """Return a sorted list of PCKs for one or more bodies.

    Parameters:
        bodies (int, list or tuple, optional): One or more SPICE body IDs; None to load
            kernels for all planetary bodies.
        name (str, optional): A SQL match string for the name of the kernel; use "%" for
            multiple wildcards and "_" for a single wildcard.
        asof (str, optional): An optional earlier date for which values should be
            returned. Wherever possible, the kernels selected will have release dates
            earlier than this date. The date is expressed as a string in ISO format or as
            a number of seconds TAI elapsed since January 1, 2000.
        after (str, optional): An optional date such that files originating earlier are
            not considered. The date is expressed as a string in ISO format or as a number
            of seconds TAI elapsed since January 1, 2000.
        redo (bool, optional): True to relax the 'asof' and 'after" constraints if no
            matching results are found; False to raise a ValueError instead.

    Returns:
        A sorted list of KernelInfo objects.
    """

    # Search database
    kernel_list = _query_kernels("PCK", name=name, body=bodies,
                                        asof=asof, after=after, redo=redo,
                                        limit=False)

    # Sort the kernels and return
    return _sort_kernels(kernel_list)

def select_spk(bodies, name=None, time=None, asof=None, after=None, redo=True):
    """Return a sorted list of SPKs for one or more bodies.

    Parameters:
        bodies (int, list or tuple): One or more SPICE body IDs; None to load kernels for
            all planetary bodies.
        name (str, optional): A SQL match string for the name of the kernel; use "%" for
            multiple wildcards and "_" for a single wildcard.
        time (tuple, optional): The start and stop times. Each time is expressed in either
            ISO format "yyyy-mm-ddThh:mm:ss" or as a number of seconds TAI elapsed since
            January 1, 2000. Use None to load the most recent complete set of kernels
            regardless of their time limits.
        asof (str, optional): An optional earlier date for which values should be
            returned. Wherever possible, the kernels selected will have release dates
            earlier than this date. The date is expressed as a string in ISO format or as
            a number of seconds TAI elapsed since January 1, 2000.
        after (str, optional): An optional date such that files originating earlier are
            not considered. The date is expressed as a string in ISO format or as a number
            of seconds TAI elapsed since January 1, 2000.
        redo (bool, optional): True to relax the 'asof' and 'after" constraints if no
            matching results are found; False to raise a ValueError instead.

    Returns:
        A sorted list of KernelInfo objects.
    """

    # Normalize the input
    if isinstance(bodies, numbers.Integral):
        bodies = [bodies]

    # Select the kernels
    spacecraft_only = True
    kernel_list = []
    for body in bodies:
        if body > 0:
            spacecraft_only = False
        kernel_list += _query_kernels("SPK", name=name, body=body, time=time,
                                             asof=asof, after=after, redo=redo,
                                             limit=False)

    # Remove kernels with overlapping time limits
    if time is None:
        time = (None, None)
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

    Parameters:
        ids (int, list or tuple): One or more negative SPICE body IDs for spacecrafts.
        inst (str, list or tuple, optional): One or more instrument names or
            abbreviations. None to return kernels for every instrument.
        types (str, list or tuple, optional): One or more kernel types ("IK", "FK",
            "SCLK") to return. None to return every kernel type.
        asof (str, optional): An optional earlier date for which values should be
            returned. Wherever possible, the kernels selected will have release dates
            earlier than this date. The date is expressed as a string in ISO format or as
            a number of seconds TAI elapsed since January 1, 2000.
        after (str, optional): An optional date such that files originating earlier are
            not considered. The date is expressed as a string in ISO format or as a number
            of seconds TAI elapsed since January 1, 2000.
        redo (bool, optional): True to relax the 'asof' and 'after' constraints if no
            matching results are found; False to raise a ValueError instead.

    Returns:
        A sorted list of KernelInfo objects.
    """

    # Normalize inputs
    if isinstance(ids, numbers.Integral):
        ids = [ids]
    if isinstance(inst, str):
        inst = [inst]

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

    Parameters:
        ids (int, list or tuple): One or more negative SPICE body IDs for spacecrafts.
        name (str, optional): A SQL match string for the name of the kernel; use "%" for
            multiple wildcards and "_" for a single wildcard.
        time (tuple, optional): The start and stop times. Each time is expressed in either
            ISO format "yyyy-mm-ddThh:mm:ss" or as a number of seconds TAI elapsed since
            January 1, 2000. Use None to load a complete set of C kernels.
        asof (str, optional): An optional earlier date for which values should be
            returned. Wherever possible, the kernels selected will have release dates
            earlier than this date. The date is expressed as a string in ISO format or as
            a number of seconds TAI elapsed since January 1, 2000.
        after (str, optional): An optional date such that files originating earlier are
            not considered. The date is expressed as a string in ISO format or as a number
            of seconds TAI elapsed since January 1, 2000.
        redo (bool, optional): True to relax the 'asof' and 'after" constraints if no
            matching results are found; False to raise a ValueError instead.

    Returns:
        A sorted list of KernelInfo objects.
    """

    # Normalize inputs
    if isinstance(ids, numbers.Integral):
        ids = [ids]

    # For each spacecraft...
    kernel_list = []
    for id in ids:

        # Select the C kernels
        kernel_list += _query_kernels("CK", name=name, time=time,
                                            body=id, asof=asof, after=after,
                                            limit=False)

    # Remove overlapping kernels and sort
    if time is None:
        time = ('0001-01-01', '3000-01-01')
    return _remove_overlaps(kernel_list, time[0], time[1])

def select_by_name(names, time=None):
    """Return a list of kernel objects associated with a list of names.

    Parameters:
        names (list): Kernel names, including version numbers, and optional file_no
            indices.
        time (tuple, optional): The start and stop times. Each time is expressed in either
            ISO format "yyyy-mm-ddThh:mm:ss" or as a number of seconds TAI elapsed since
            January 1, 2000. Use None to load all the matching kernels.
    """

    # Search database
    kernel_list = _query_by_name(names, time)

    # Sort the kernels
    return _sort_kernels(kernel_list)

def select_by_filespec(filespecs, time=None):
    """Return a list of kernel objects associated with a list of names.

    Parameters:
        names (list): A list of file specifications or match patterns. The file
            specification need not contain the directory path.
        time (tuple, optional): The start and stop times. Each time is expressed in either
            ISO format "yyyy-mm-ddThh:mm:ss" or as a number of seconds TAI elapsed since
            January 1, 2000. Use None to load all the matching kernels.
    """

    # Search database, DO NOT sort!
    return _query_by_filespec(filespecs, time)

##########################################################################################
# Public API for returning text kernels as dictionaries
##########################################################################################

def as_dict(kernel_list):
    """Return a dictionary containing the information in text kernels.

    Binary kernels are ignored.
    """

    pfx = get_spice_filecache_prefix()

    result = {}
    for kernel in kernel_list:

        # Check for a text kernel
        ext = os.path.splitext(kernel.filespec)[1].lower()
        if ext[0:2] != ".t":
            continue

        local_path = pfx.retrieve(kernel.filespec)
        result = textkernel.from_file(local_path, tkdict=result)

    return result

##########################################################################################
# Public API for furnishing kernels
##########################################################################################

def furnish_kernels(kernel_list, fast=True):
    """Furnish a pre-sorted list of kernels for use by the cspyce module.

    Parameters:
        fast (bool, optional): True to skip the loading kernels that have already been
            loaded. False to unload and load them again, thereby raising their priority.

    Returns:
        An ordered list of the names, versions and file_nos of the kernels loaded.
            This can be used to re-load the exact same selection of kernels again at a
            later date.
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

    pfx = get_spice_filecache_prefix()

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
        if TRANSLATOR:
            new_filepaths = []
            for oldpath in filepaths:
                newpath = TRANSLATOR(oldpath)
                if newpath:
                    new_filepaths.append(newpath)

            filepaths = new_filepaths

        abspaths = pfx.retrieve(filepaths, exception_on_fail=False)

        for (abspath, filepath) in zip(abspaths, filepaths):
            # Save the info for each furnished file if it exists.
            if not isinstance(abspath, Path):
                if DEBUG:
                    abspath = pfx.get_local_path(filepath, create_parents=False)
                else:
                    warnings.warn(f'SPICE kernel not found: {pfx}/{filepath}',
                                  RuntimeWarning)
                    continue

            # Remove the name from earlier in the list if necessary
            if abspath in abspath_list:
                abspath_list.remove(abspath)

            # Always add it at the end
            abspath_list.append(abspath)
            abspath_types[abspath] = kernel.kernel_type     # track kernel types

            basename = os.path.basename(abspath)
            if basename in FURNISHED_INFO:
                if kernel not in FURNISHED_INFO[basename]:
                    FURNISHED_INFO[basename].append(kernel)
            else:
                FURNISHED_INFO[basename] = [kernel]

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
                if fast:
                    continue
                furnished_names.remove(name)

            furnished_names.append(name)

    # Append file number ranges into the names in the list returned
    for (name,filenos) in fileno_dict.items():
        k = name_list.index(name)
        name_list[k] = name + _fileno_str(filenos)

        # Track kernels loaded by file_no
        if not DEBUG:
            if name not in FURNISHED_FILENOS:
                FURNISHED_FILENOS[name] = []

            fileno_list = FURNISHED_FILENOS[name]
            for fileno in filenos:
                if fileno in fileno_list:
                    if fast:
                        continue
                    fileno_list.remove(fileno)

                fileno_list.append(fileno)

    return name_list

def furnish_lsk(asof=None, after=None, redo=True, fast=True):
    """Furnish selected leapseconds kernels and return a list of names.

    Parameters:
        asof (str, optional): An optional earlier date for which values should be
            returned. Wherever possible, the kernels selected will have release dates
            earlier than this date. The date is expressed as a string in ISO format or as
            a number of seconds TAI elapsed since January 1, 2000.
        after (str, optional): An optional date such that files originating earlier are
            not considered. The date is expressed as a string in ISO format or as a number
            of seconds TAI elapsed since January 1, 2000.
        redo (bool, optional): True to relax the 'asof' and 'after" constraints if no
            matching results are found; False to raise a ValueError instead.
        fast (bool, optional): True to skip the loading kernels that have already been
            loaded. False to unload and load them again, thereby raising their priority.

    Returns:
        (list): A list of kernel names in load order.
    """

    # Search the database
    kernel_list = select_lsk(asof=asof, after=after, redo=redo)

    # Load the kernels and return the names
    return furnish_kernels(kernel_list, fast=fast)

def furnish_pck(bodies=None, name=None, asof=None, after=None, redo=True,
                fast=True):
    """Furnish selected PCKs for one or more bodies.

    Parameters:
        bodies (int, list or tuple, optional): One or more SPICE body IDs; None to load
            kernels for all planetary bodies.
        name (str, optional): A SQL match string for the name of the kernel; use "%" for
            multiple wildcards and "_" for a single wildcard.
        asof (str, optional): An optional earlier date for which values should be
            returned. Wherever possible, the kernels selected will have release dates
            earlier than this date. The date is expressed as a string in ISO format or as
            a number of seconds TAI elapsed since January 1, 2000.
        after (str, optional): An optional date such that files originating earlier are
            not considered. The date is expressed as a string in ISO format or as a number
            of seconds TAI elapsed since January 1, 2000.
        redo (bool, optional): True to relax the 'asof' and 'after" constraints if no
            matching results are found; False to raise a ValueError instead.
        fast (bool, optional): True to skip the loading kernels that have already been
            loaded. False to unload and load them again, thereby raising their priority.

    Returns:
        (list): A list of kernel names in load order.
    """

    # Search database
    kernel_list = select_pck(bodies=bodies, name=name,
                             asof=asof, after=after, redo=redo)

    # Load the kernels and return the names
    return furnish_kernels(kernel_list, fast=fast)

def furnish_spk(bodies, name=None, time=None, asof=None, after=None, redo=True,
                fast=True):
    """Furnish SPKs for one or more bodies and spacecrafts.

    Parameters:
        bodies (int, list or tuple): One or more SPICE body IDs; None to load kernels for
            all planetary bodies.
        name (str, optional): A SQL match string for the name of the kernel; use "%" for
            multiple wildcards and "_" for a single wildcard.
        time (tuple, optional): The start and stop times. Each time is expressed in either
            ISO format "yyyy-mm-ddThh:mm:ss" or as a number of seconds TAI elapsed since
            January 1, 2000. Use None to load the most recent complete set of kernels
            regardless of their time limits.
        asof (str, optional): An optional earlier date for which values should be
            returned. Wherever possible, the kernels selected will have release dates
            earlier than this date. The date is expressed as a string in ISO format or as
            a number of seconds TAI elapsed since January 1, 2000.
        after (str, optional): An optional date such that files originating earlier are
            not considered. The date is expressed as a string in ISO format or as a number
            of seconds TAI elapsed since January 1, 2000.
        redo (bool, optional): True to relax the 'asof' and 'after" constraints if no
            matching results are found; False to raise a ValueError instead.
        fast (bool, optional): True to skip the loading kernels that have already been
            loaded. False to unload and load them again, thereby raising their priority.

    Returns:
        (list): A list of kernel names in load order.
    """

    # Search database
    kernel_list = select_spk(bodies, name=name, time=time, asof=asof,
                             after=after, redo=redo)

    # Furnish the kernels and return the names
    return furnish_kernels(kernel_list, fast=fast)

def furnish_inst(ids, inst=None, types=None, asof=None, after=None, redo=True,
                      fast=True):
    """Furnish IKs, FKs and SCLKs for one or more spacecrafts and instruments.

    Parameters:
        ids (int, list or tuple): One or more negative SPICE body IDs for spacecrafts.
        inst (str, list or tuple, optional): One or more instrument names or
            abbreviations. None to furnish kernels for every instrument.
        types (str, list or tuple, optional): One or more kernel types ("IK", "FK",
            "SCLK") to furnish. None to return every kernel type.
        asof (str, optional): An optional earlier date for which values should be
            returned. Wherever possible, the kernels selected will have release dates
            earlier than this date. The date is expressed as a string in ISO format or as
            a number of seconds TAI elapsed since January 1, 2000.
        after (str, optional): An optional date such that files originating earlier are
            not considered. The date is expressed as a string in ISO format or as a number
            of seconds TAI elapsed since January 1, 2000.
        redo (bool, optional): True to relax the 'asof' and 'after" constraints if no
            matching results are found; False to raise a ValueError instead.

    Returns:
        (list): A list of kernel names in load order.
    """

    # Search database
    kernel_list = select_inst(ids, inst, types, asof, after, redo)

    # Furnish the kernels and return the names
    return furnish_kernels(kernel_list, fast=fast)

def furnish_ck(ids, name=None, time=None, asof=None, after=None, redo=True,
                    fast=True):
    """Furnish CKs for one or more spacecrafts.

    Parameters:
        ids (int, list or tuple): One or more negative SPICE body IDs for spacecrafts.
        name (str, optional): A SQL match string for the name of the kernel; use "%" for
            multiple wildcards and "_" for a single wildcard.
        time (tuple, optional): The start and stop times. Each time is expressed in either
            ISO format "yyyy-mm-ddThh:mm:ss" or as a number of seconds TAI elapsed since
            January 1, 2000. Use None to load a complete set of C kernels.
        asof (str, optional): An optional earlier date for which values should be
            returned. Wherever possible, the kernels selected will have release dates
            earlier than this date. The date is expressed as a string in ISO format or as
            a number of seconds TAI elapsed since January 1, 2000.
        after (str, optional): An optional date such that files originating earlier are
            not considered. The date is expressed as a string in ISO format or as a number
            of seconds TAI elapsed since January 1, 2000.
        redo (bool, optional): True to relax the 'asof' and 'after" constraints if no
            matching results are found; False to raise a ValueError instead.
        fast (bool, optional): True to skip the loading kernels that have already been
            loaded. False to unload and load them again, thereby raising their priority.

    Returns:
        (list): A list of kernel names in load order.
    """

    # Search database
    kernel_list = select_ck(ids, name=name, time=time,
                            asof=asof, after=after, redo=redo)

    # Furnish the kernels and return the names
    return furnish_kernels(kernel_list, fast=fast)

def furnish_by_name(names, time=None, fast=True):
    """Furnish kernels identified by a list of names.

    Parameters:
        names (list): Kernel names, including version numbers, and optional file_no
            indices.
        time (tuple, optional): The start and stop times. Each time is expressed in either
            ISO format "yyyy-mm-ddThh:mm:ss" or as a number of seconds TAI elapsed since
            January 1, 2000. Use None to load all the matching kernels.
        fast (bool, optional): True to skip the loading kernels that have already been
            loaded. False to unload and load them again, thereby raising their priority.

    Returns:
        (list): A list of kernel names in load order. This will typically match the input
            names unless different time limits are applied.
    """

    # Search database
    kernel_list = select_by_name(names, time)

    # Furnish the kernels and return the names
    return furnish_kernels(kernel_list, fast=fast)

def furnish_by_metafile(metafile, time=None, asof=None):
    """Furnish kernels identified by the path to a metakernel.

    Parameters:
        metafile (str or FCPath): A file path to a metafile, or the name of a metafile in
            the SPICE database, or the filespec of a meta kernel in the SPICE database.
        time (tuple, optional): Consisting of a start and stop time, each expressed as a
            string in ISO format, "yyyy-mm-ddThh:mm:ss". Alternatively, times may be given
            as elapsed seconds TAI since January 1, 2000. Use None to return kernels
            regardless of the time.
        asof (str, optional): An optional date earlier than today for which values should
            be returned. Wherever possible, the kernels selected will have release dates
            earlier than this date. The date is expressed as a string in ISO format or as
            a number of seconds TAI elapsed since January 1, 2000.

    Returns:
        (list): A list of kernel names in load order.
    """

    pfx = get_spice_filecache_prefix()

    # Search database
    kernel_names = []
    if not os.path.exists(metafile):
        spice_path = get_spice_path()
        try:
            kernel_list = _query_kernels('META', name=metafile, asof=asof)
            metafile = pfx.retrieve(kernel_list[-1].filespec)
            kernel_names = [kernel_list[-1].full_name]
        except ValueError:
            kernel_list = _query_kernels('META', path=metafile, asof=asof)
            metafile = pfx.retrieve(kernel_list[-1].filespec)
            kernel_names = [kernel_list[-1].full_name]

    local_path = get_spice_filecache.retrieve(metafile)
    filespecs = textkernel.from_file(local_path)['KERNELS_TO_LOAD']

    kernel_list = select_by_filespec(filespecs, time=time)

    # Furnish the kernels and return the names
    return furnish_kernels(kernel_list, fast=False) + kernel_names

def furnish_by_filepath(filepath):
    """Furnish a file by its full file path. This file need not be in the
    database.
    """

    kernels = kernels_from_filespec(filepath)
    furnish_kernels(kernels, fast=False)

##########################################################################################
# Public API for unloading kernels
##########################################################################################

def unload_by_name(names):
    """Unload kernels based on a list of kernel names."""

    global FURNISHED_ABSPATHS, FURNISHED_NAMES, FURNISHED_INFO
    global FURNISHED_FILENOS

    # Search database
    kernel_list = _query_by_name(names)

    # Sort the kernels
    kernel_list = _sort_kernels(kernel_list)

    # For each kernel...
    pfx = get_spice_filecache_prefix()

    for kernel in kernel_list:
        key = kernel.kernel_type

        # Remove the kernel files from the dictionary and unload from SPICE
        filespecs = kernel.filespec.split(',')
        for filespec in filespecs:
            abspath = pfx.get_local_path(filespec)
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

    # For each selected type...
    for key in types:

        # Unload each file from SPICE
        abspath_list = FURNISHED_ABSPATHS[key]
        for abspath in abspath_list:
            cspyce.unload(abspath)
            del FURNISHED_INFO[os.path.basename(abspath)]

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
    database.
    """

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

def unload_all():
    """Unload all SPICE kernels."""

    global FURNISHED_NAMES, FURNISHED_FILENOS, FURNISHED_INFO

    for ktype, abspaths in FURNISHED_ABSPATHS.items():
        for abspath in abspaths:
            cspyce.unload(abspath)
        FURNISHED_ABSPATHS[ktype] = []
        FURNISHED_NAMES[ktype] = []

    FURNISHED_FILENOS = {}
    FURNISHED_INFO = {}

##########################################################################################
# Public API for names of kernels
##########################################################################################

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
        if kernel.file_no is None:
            continue

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
    kernels.
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
        for filespec in FURNISHED_ABSPATHS[key]:
            basename = os.path.basename(filespec)
            name_list.append(basename)

    return name_list

def used_basenames(types=[], time=None, bodies=[], sc=None, inst=None,
                             slop=6*60*60):
    """Return a list of SPICE file basenames needed for a particular list of
    bodies and frames at a particular time.
    """

    global FURNISHED_NAMES, FURNISHED_FILENOS
    global KERNEL_TYPE_SORT_ORDER

    # Normalize input
    if types is None or types == []:
        types = KERNEL_TYPE_SORT_ORDER
    elif type(types) == str:
        types = [types]

    # Normalize time
    if time is not None:
        if isinstance(time, (str, numbers.Real)):
            time = [time, time]

        time_tai = []
        for tval in time:
            if isinstance(tval, str):
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
      if key == 'CK' and not ck_needed:
        continue
      if key == 'IK' and not inst:
        continue

      # Walk down list
      temp_list = []
      for filespec in FURNISHED_ABSPATHS[key]:
        basename = os.path.basename(filespec)
        used = False
        for info in FURNISHED_INFO[basename]:

            if time and info.start_time:
                if time[1] < info.start_tai - slop:
                    continue
                if time[0] > info.stop_tai  + slop:
                    continue

            if bodies and info.spice_id:
                if info.spice_id not in bodies:
                    continue

            used = True

        if used:
            temp_list.append(basename)

      if key == 'IK':
        reduced_list = [name for name in temp_list if inst in name.lower()]
        if reduced_list:
            temp_list = reduced_list

      basename_list += temp_list

    return basename_list

##########################################################################################
# DEPRECATED: Special kernel loader for Cassini
# Deleted 2/22/2020
##########################################################################################

def furnish_cassini_kernels(start_time, stop_time, instrument=None, asof=None):
    """A routine designed to load all needed SPICE kernels for a SPICE calculation
    involving the Cassini spacecraft.

    Parameters:
        start_time (str): The start time of the period of interest, in ISO format,
            "yyyy-mm-ddThh:mm:ss".
        stop_time (str): The stop time of the period of interest.
        instrument (list, optional): Instruments to be used. If the list is empty, C
            kernels will not be loaded. If one or more instruments are listed, the C
            kernels and needed Frames kernels will be loaded. Options are the standard
            mission abbreviations, e.g., "ISS", "VIMS", "CIRS", "UVIS", etc.
        asof (str, optional): An optional earlier date for which values should be
            returned. Wherever possible, the kernels selected will have release dates
            earlier than this date. The date is expressed as a string in ISO format.

    Returns:
        (list): The names of all the kernels loaded.
    """

    names = []

    # Leapseconds Kernel (LSK)
    names += furnish_lsk(asof=asof)

    # Instruments and frames
    names += furnish_inst(-82, instrument, asof=asof)

    # Planetary Constants
    bodies = [699] + lrange(601,654) + [65035, 65040, 65041] # plus a few more
    names += furnish_pck(bodies, asof=asof)

    # Ephemerides (SP Kernels)
    names += furnish_spk(bodies + [-82], time=(start_time,stop_time), asof=asof)

    # C (pointing) Kernels
    names += furnish_ck(-82, time=(start_time, stop_time), asof=asof)

    return names

##########################################################################################
# Special kernel loader for every planet and moon
##########################################################################################

def furnish_solar_system(start_time=None, stop_time=None, asof=None,
                         planets=(1,2,3,4,5,6,7,8,9)):
    """A routine designed to load all the SPK, FK and planetary constants files needed for
    the planets and moons of the Solar System.

    Parameters:
        start_time (str, optional): The start time of the period of interest, in ISO
            format, "yyyy-mm-ddThh:mm:ss" or in seconds TAI past January 1, 2000. Use None
            to furnish the latest kernels irrespective of their time limits.
        stop_time (str, optional): The stop time of the period of interest.
        asof (str, optional): An optional earlier date for which values should be
            returned. Wherever possible, the kernels selected will have release dates
            earlier than this date. The date is expressed as a string in ISO format.
        planets (int, optional): 1-9 to load kernels for a particular planet and its
            moons. 0 or None to load nine planets (including Pluto). Use a tuple to list
            more than one planet number.

    Returns:
        (list): The names of all the kernels loaded.
    """

    if planets is None or planets == 0:
        planets = (1,2,3,4,5,6,7,8,9)
    if isinstance(planets, numbers.Integral):
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

##########################################################################################
