# oops/instrument/__init__.py

import unittest

import oops
import oops.instrument.hst

def from_file(file_spec):
    """Given the name of a file, this function returns an associated Observation
    object describing the data found in the file."""

    # Confirm that the file exists
    f = open(file_spec)
    f.close()

    ####################################
    # HST case
    ####################################

    # See if this is an HST file
    try:
        return oops.instrument.hst.from_file(file_spec)

    # A RuntimeError means this file is from HST but is not supported

    # An IOError means this is not an HST file
    except IOError: pass

    ####################################
    # Unrecognized case
    ####################################

    raise IOError("unidentified instrument host: " + file_spec)

################################################################################
# UNIT TESTS
################################################################################

class Test_Instrument(unittest.TestCase):

    def runTest(self):

        # TBD

        pass

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
