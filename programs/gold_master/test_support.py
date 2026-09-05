##########################################################################################
# programs/gold_master/test_support.py
##########################################################################################
"""File paths used by the oops tests and by the gold master tests.

Defines the locations of the test data, the SPICE kernels, the gold master files, and the
directory to which newly generated backplane arrays are written, based on the environment
variables `OOPS_RESOURCES`, `OOPS_TEST_DATA_PATH`, `OOPS_GOLD_MASTER_PATH`, and
`OOPS_BACKPLANE_OUTPUT_PATH`. Any of these may name a cloud resource such as
"gs://rms-oops-resources/gold_master".
"""

import os

from filecache import FileCache

__all__ = ['OOPS_RESOURCES', 'OOPS_TEST_DATA_PATH', 'OOPS_GOLD_MASTER_PATH',
           'OOPS_BACKPLANE_OUTPUT_PATH', 'TEST_DATA_FILECACHE', 'TEST_SPICE_FILECACHE',
           'TEST_DATA_PREFIX', 'TEST_SPICE_PREFIX', 'GOLD_MASTER_PREFIX',
           'BACKPLANE_OUTPUT_PREFIX']

# import filecache
# filecache.set_easy_logger()

# Environment variables used to support oops and oops host testing:
#
# - $OOPS_RESOURCES is the top-level directory. This module uses three of its
#   subdirectories, "test_data", "SPICE", and "gold_master"; see the README for the full
#   set.
# - $OOPS_TEST_DATA_PATH will override the location of the "test_data" directory.
# - $OOPS_GOLD_MASTER_PATH will override the location of the "gold_master" directory.
# - $OOPS_BACKPLANE_OUTPUT_PATH specifies the location in which generated backplanes
#   should be written.
#
# Any environment variable may be a URI for a cloud resource such as
#   gs://rms-oops-resources

try:
    OOPS_RESOURCES = os.environ['OOPS_RESOURCES']
except KeyError:
    OOPS_RESOURCES = None

try:
    OOPS_TEST_DATA_PATH = os.environ['OOPS_TEST_DATA_PATH']
except KeyError:
    if OOPS_RESOURCES:
        OOPS_TEST_DATA_PATH = f'{OOPS_RESOURCES}/test_data'
    else:
        OOPS_TEST_DATA_PATH = None

try:
    OOPS_GOLD_MASTER_PATH = os.environ['OOPS_GOLD_MASTER_PATH']
except KeyError:
    if OOPS_RESOURCES:
        OOPS_GOLD_MASTER_PATH = f'{OOPS_RESOURCES}/gold_master'
    else:
        OOPS_GOLD_MASTER_PATH = None

try:
    OOPS_BACKPLANE_OUTPUT_PATH = os.environ['OOPS_BACKPLANE_OUTPUT_PATH']
except KeyError:
    OOPS_BACKPLANE_OUTPUT_PATH = os.getcwd()


# The FileCache in which to store the "$OOPS_RESOURCES/test_data" directory
TEST_DATA_FILECACHE = FileCache('oops_test_data')

# The FileCache in which to store the "$OOPS_TEST_DATA_PATH/SPICE" directory; this should
# be DIFFERENT from the name used in spicedb because these could be different kernels with
# the same name
TEST_SPICE_FILECACHE = FileCache('oops_test_kernels')

if OOPS_TEST_DATA_PATH:
    TEST_DATA_PREFIX = TEST_DATA_FILECACHE.new_path(OOPS_TEST_DATA_PATH)
    TEST_SPICE_PREFIX = TEST_SPICE_FILECACHE.new_path(
        f'{OOPS_TEST_DATA_PATH}/SPICE')
else:
    TEST_DATA_PREFIX = None
    TEST_SPICE_PREFIX = None

if OOPS_GOLD_MASTER_PATH:
    GOLD_MASTER_PREFIX = TEST_DATA_FILECACHE.new_path(OOPS_GOLD_MASTER_PATH)
else:
    GOLD_MASTER_PREFIX = None

BACKPLANE_OUTPUT_PREFIX = TEST_DATA_FILECACHE.new_path(
    OOPS_BACKPLANE_OUTPUT_PATH)

##########################################################################################
