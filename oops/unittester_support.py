################################################################################
# oops/oops_resources.py
################################################################################

import os

from filecache import FileCache

# Environment variables used to support oops and oops host testing:
#
# - $OOPS_RESOURCES is the top-level directory. It is expected to have two
#   subdirectories, "test_data" and "SPICE".
# - $OOPS_TEST_DATA_PATH will override the location of the "test_data"
#   directory.
#
# Either environment variable may be a URI for a cloud resource such as
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


# The FileCache in which to store the "$OOPS_RESOURCES/test_data" directory
TEST_DATA_FILECACHE = FileCache(shared='oops_test_data')

# The FileCache in which to store the "$OOPS_RESOURCES/SPICE" directory; this
# should be DIFFERENT from the name used in spicedb because these could
# be different kernels with the same name
TEST_SPICE_FILECACHE = FileCache(shared='oops_test_kernels')

if OOPS_TEST_DATA_PATH:
    TEST_DATA_PREFIX = TEST_DATA_FILECACHE.new_prefix(OOPS_TEST_DATA_PATH)
    TEST_SPICE_PREFIX = TEST_SPICE_FILECACHE.new_prefix(f'{OOPS_TEST_DATA_PATH}/SPICE')
else:
    TEST_DATA_PREFIX = None
    TEST_SPICE_PREFIX = None

################################################################################
