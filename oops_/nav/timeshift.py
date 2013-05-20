################################################################################
# oops_/nav/timeshift.py: Subclass TimeShift of class Navigation.
#
# 5/21/12 MRS - Created but untested.
################################################################################

import numpy as np

from oops_.array.all import *
from oops_.nav.navigation_ import Navigation

class TimeShift(Navigation):
    """A TimeShift is a Navigation subclass that shifts all of the times
    associated with an observation.
    """

    def __init__(self, secs):
        """Constructor for a TimeShift object.

        Input:
            secs        a single constant in seconds to be added to all times,
                        represented by a constant or a single-element
                        tuple/list/array.
        """

        self.nparams = 1
        self.set_params(secs)

    ####################################

    def set_params(self, secs):
        """Part of the Fittable interface. Re-defines the navigation given a
        new set of parameters."""

        # Check the parameters
        if np.shape(secs) == ():
            secs = np.array([secs])
        else:
            secs = np.array(secs)

        assert secs.shape == (self.nparams,)

        # Save the parameter
        self.secs = secs

    ####################################

    def get_params(self):
        """Part of the Fittable interface. Returns the current parameters.
        """

        return self.secs

    ####################################

    def copy(self):
        """Part of the Fittable interface. Returns a deep copy of the object.
        """

        return TimeShift(self.secs)

    ####################################

    def shift(self, time, uv, partials=False):
        """Applies the time-shift to the observation. This should be applied to
        to times defined by external events, returning times applicable within
        the span of the observation.

        Input:
            time        a Scalar of external event times.
            uv          a Pair defining the (u,v) coordinates at which the
                        time-shift applies.
            partials    if True, then the partial derivatives with respect to
                        the parameters are also returned as a subfield "d_dnav".
                        The partials are represented by a MatrixN with item
                        shape [1,N], where N is the number of parameters.

        Return:         a Vector3 of times within the span of the observation.
        """

        time = Scalar.as_scalar(time) + secs

        if partials:
            time.insert_subfield("d_dnav", Scalar(1.))

        return time

    ####################################

    def unshift(self, time, uv, partials=False):
        """Removes the time-shift from an observation's timing. This should be
        applied to observation event times, returning times applicable to
        external events.

        Input:
            time        a Scalar of observation event times.
            uv          a Pair defining the (u,v) coordinates at which the
                        time-shift applies.
            partials    if True, then the partial derivatives with respect to
                        the parameters are also returned as a subfield "d_dnav".
                        The partials are represented by a MatrixN with item
                        shape [1,N], where N is the number of parameters.

        Return:         a Vector3 of times applicable to external events.
        """

        time = Scalar.as_scalar(time) - secs

        if partials:
            time.insert_subfield("d_dnav", Scalar(-1.))

        return time

################################################################################
# UNIT TESTS
################################################################################

import unittest
import numpy.random as random

class Test_TimeShift(unittest.TestCase):

    def runTest(self):

        # TBD, testing needed!
        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
