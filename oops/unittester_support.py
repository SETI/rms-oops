################################################################################
# oops/unittester_support.py
################################################################################

import os

# Attributes ending with underscore contain a trailing "/"; others do not

try:
    OOPS_RESOURCES = os.environ['OOPS_RESOURCES']
except KeyError:
    OOPS_RESOURCES = ''
    OOPS_RESOURCES_ = ''
else:
    OOPS_RESOURCES = os.path.normpath(OOPS_RESOURCES)
    OOPS_RESOURCES = OOPS_RESOURCES.rstrip('/')
    OOPS_RESOURCES_ = OOPS_RESOURCES + '/'

# test_data directory, converted to a physical path
# Default is $OOPS_RESOURCES/test_data, but this can be changed by the user
# by defining the environment variable OOPS_TEST_DATA_PATH.

try:
    OOPS_TEST_DATA_PATH = os.environ['OOPS_TEST_DATA_PATH']
except KeyError:
    if OOPS_RESOURCES:
        OOPS_TEST_DATA_PATH = os.path.join(OOPS_RESOURCES, 'test_data')
        OOPS_TEST_DATA_PATH_ = OOPS_TEST_DATA_PATH + '/'
    else:
        OOPS_TEST_DATA_PATH = ''
        OOPS_TEST_DATA_PATH_ = ''
else:
    OOPS_TEST_DATA_PATH = os.path.realpath(OOPS_TEST_DATA_PATH)
    OOPS_TEST_DATA_PATH = os.path.abspath(OOPS_TEST_DATA_PATH)
    OOPS_TEST_DATA_PATH = OOPS_TEST_DATA_PATH.rstrip('/')
    OOPS_TEST_DATA_PATH_ = OOPS_TEST_DATA_PATH + '/'

# Alternative global variable name for the same directory
TESTDATA_PARENT_DIRECTORY  = OOPS_TEST_DATA_PATH
TESTDATA_PARENT_DIRECTORY_ = OOPS_TEST_DATA_PATH_

# gold_master directory, converted to a physical path
# Default is $OOPS_RESOURCES/gold_master, but this can be changed by the user
# by defining the environment variable OOPS_GOLD_MASTER_PATH.

try:
    OOPS_GOLD_MASTER_PATH = os.environ['OOPS_GOLD_MASTER_PATH']
except KeyError:
    if OOPS_RESOURCES:
        OOPS_GOLD_MASTER_PATH = os.path.join(OOPS_RESOURCES, 'gold_master')
        OOPS_GOLD_MASTER_PATH_ = OOPS_GOLD_MASTER_PATH + '/'
    else:
        OOPS_GOLD_MASTER_PATH = ''
        OOPS_GOLD_MASTER_PATH_ = ''
else:
    OOPS_GOLD_MASTER_PATH = os.path.realpath(OOPS_GOLD_MASTER_PATH)
    OOPS_GOLD_MASTER_PATH = os.path.abspath(OOPS_GOLD_MASTER_PATH)
    OOPS_GOLD_MASTER_PATH = OOPS_GOLD_MASTER_PATH.rstrip('/')
    OOPS_GOLD_MASTER_PATH_ = OOPS_GOLD_MASTER_PATH + '/'

# Local user path for backplane testing
# This is defined by the environment variable OOPS_BACKPLANE_OUTPUT_PATH.
try:
    OOPS_BACKPLANE_OUTPUT_PATH = os.environ['OOPS_BACKPLANE_OUTPUT_PATH']
except KeyError:
    OOPS_BACKPLANE_OUTPUT_PATH = ''
    OOPS_BACKPLANE_OUTPUT_PATH_ = ''
else:
    OOPS_BACKPLANE_OUTPUT_PATH = os.path.normpath(OOPS_BACKPLANE_OUTPUT_PATH)
    OOPS_BACKPLANE_OUTPUT_PATH = OOPS_BACKPLANE_OUTPUT_PATH.rstrip('/')
    OOPS_BACKPLANE_OUTPUT_PATH_ = OOPS_BACKPLANE_OUTPUT_PATH + '/'

