################################################################################
# oops/tools.py
#
# 2/6/12 Created (MRS) - Based on parts of oops.py.
# 2/18/12 Modified (MRS) - Moved define_solar_system to body.py.
################################################################################

import spicedb
import julian
import os

################################################################################
# Useful utilities
################################################################################

LSK_LOADED = False

def load_leap_seconds():
    """Loads the most recent leap seconds kernel if it was not already loaded.
    """

    global LSK_LOADED

    if LSK_LOADED: return

    # Query for the most recent LSK
    spicedb.open_db()
    lsk = spicedb.select_kernels("LSK")
    spicedb.close_db()

    # Furnish the LSK to the SPICE toolkit
    spicedb.furnish_kernels(lsk)

    # Initialize the Julian toolkit
    julian.load_from_kernel(os.path.join(spicedb.get_spice_path(),
                                         lsk[0].filespec))

    LSK_LOADED = True

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_tools(unittest.TestCase):

    def runTest(self):

        # TBD
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
