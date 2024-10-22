################################################################################
# oops/gold_master/test_support.py
################################################################################

import os
from oops.unittester_support import (OOPS_RESOURCES,
                                     TEST_DATA_FILECACHE,
                                     TEST_DATA_PREFIX)

# import filecache
# filecache.set_easy_logger()

# Environment variables used to support oops "gold master" testing:
#
# - $OOPS_RESOURCES is the top-level directory. It is expected to have the
#   subdirectory "gold_master".
# - $OOPS_GOLD_MASTER_PATH will override the location of the "gold_master"
#   directory.
# - $OOPS_BACKPLANE_OUTPUT_PATH specifies the location in which generated
#   backplanes should be written.
#
# Any environment variable may be a URI for a cloud resource such as
#   gs://rms-oops-resources/gold_master

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


if OOPS_GOLD_MASTER_PATH:
    GOLD_MASTER_PREFIX = TEST_DATA_FILECACHE.new_prefix(OOPS_GOLD_MASTER_PATH)
else:
    GOLD_MASTER_PREFIX = None

BACKPLANE_OUTPUT_PREFIX = TEST_DATA_FILECACHE.new_prefix(
    OOPS_BACKPLANE_OUTPUT_PATH)

################################################################################
