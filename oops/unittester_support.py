################################################################################
# oops/unittester_support.py
#
# 1/27/14 - Created by RSF
################################################################################

import os

try:
    TESTDATA_PARENT_DIRECTORY = os.environ["OOPS_TEST_DATA"]
except KeyError:
    TESTDATA_PARENT_DIRECTORY = ''
