################################################################################
# oops/unittester_support.py
################################################################################

import os

try:
    TESTDATA_PARENT_DIRECTORY = os.environ["OOPS_TEST_DATA"]
except KeyError:
    TESTDATA_PARENT_DIRECTORY = ''
