################################################################################
# oops_/nav/nullnav.py: Subclass NullNav of class Navigation
#
# 5/21/12 MRS - Created.
################################################################################

import numpy as np

from oops_.array.all import *
from oops_.nav.navigation_ import Navigation

class NullNav(Navigation):
    """This defines a null navigation. It is a subclass of Fittable but has no
    free parameters. Line-of-sight vectors and times are returned unchanged.
    """

    def __init__(self, angles):
        """Constructor for a NullNav object. """

        self.nparams = 0
        self.params = np.array(())

        self.dlos_dparams = MatrixN(np.zeros((3,0)))
        self.dtime_dparams = MatrixN(np.zeros((1,0)))

    ####################################

    def set_params(self, params):
        """Part of the Fittable interface. Re-defines the navigation given a
        new set of parameters."""

        assert self.params.shape == (self.nparams,)

    ####################################

    def get_params(self):
        """Part of the Fittable interface. Returns the current parameters.
        """

        return self.params

    ####################################

    def copy(self):
        """Part of the Fittable interface. Returns a deep copy of the object.
        """

        return NullNav()

    # Remaining methods are just the defaults defined in navigation.py

################################################################################
# UNIT TESTS
################################################################################

import unittest
import numpy.random as random

class Test_NullNav(unittest.TestCase):

    def runTest(self):

        # TDB, but not much to test
        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
