# oops/instrument/__init__.py

import numpy as np
import pylab
import unittest

import oops

################################################################################
# Instrument Class
################################################################################

class Instrument(object):
    """An Instrument is an abstract class that interprets a given data file and
    returns the associated Observation object. Instrument classes have no
    instances but use inheritance to define the relationships between different
    missions and instruments.
    """

    pass

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
