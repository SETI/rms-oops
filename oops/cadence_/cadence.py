################################################################################
# oops/cadence_/cadence.py: Abstract class Cadence
################################################################################

import numpy as np
from polymath import *

class Cadence(object):
    """Cadence is an abstract class that defines the timing of an observation.

    At minimum, these attributes are required:
        time            a tuple or Pair defining the start time and end time of
                        the observation overall, in seconds TDB.
        midtime         the mid-time of the observation, in seconds TDB.
        lasttime        the start-time of the last time step, in seconds TDB.
        shape           a tuple of integers defining the shape of the indices.
        is_continuous   True if the cadence contains no gaps in time between
                        the start and end.
    """

########################################################
# Methods to be defined for each Cadence subclass
########################################################

    def __init__(self):
        """A constructor."""

        pass

    def time_at_tstep(self, tstep, mask=True):
        """Returns the time associated with the given time step. This method
        supports non-integer step values.

        Input:
            tstep       a Scalar time step index or a Pair of indices.
            mask        True to mask values outside the time limits.

        Return:         a Scalar of times in seconds TDB.
        """

        raise NotImplementedException("time_at_tstep() is not implemented")

    def time_range_at_tstep(self, tstep, mask=True):
        """Returns the range of time associated with the given integer time
        step index.

        Input:
            indices     a Scalar time step index or a Pair of indices.
            mask        True to mask values outside the time limits.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        raise NotImplementedException("time_range_at_tstep() " +
                                      "is not implemented")

    def tstep_at_time(self, time, mask=True):
        """Returns a the Scalar time step index or a Pair of indices
        associated with a time in seconds TDB.

        Input:
            time        a Scalar of times in seconds TDB.
            mask        True to mask time values not sampled within the cadence.

        Return:         a Scalar or Pair of time step indices.
        """

        raise NotImplementedException("tstep_at_time() is not implemented")

    def time_shift(self, secs):
        """Returns a duplicate of the given cadence, with all times shifted by
        a specified number of seconds."

        Input:
            secs        the number of seconds to shift the time later.
        """

        raise NotImplementedException("time_shift() is not implemented")

    def as_continuous(self):
        """Returns a shallow copy of the given cadence, with equivalent strides
        but with the property that the cadence is continuous.
        """

        raise NotImplementedException("as_continuous() is not implemented")

    ####################################################
    # Methods probably not requiring overrides
    ####################################################

    def time_is_inside(self, time, inclusive=True):
        """Returns a boolean Numpy array indicating which elements in a given
        Scalar of times fall inside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to include the end moment of a time interval;
                        False to exclude.

        Return:         a boolean Numpy array indicating which time values are
                        sampled by the cadence.
        """

        # Default behavior is to include all times between start and stop
        if inclusive:
            return (time.vals >= self.time[0]) & (time.vals <= self.time[1])
        else:
            return (time.vals >= self.time[0]) & (time.vals < self.time[1])

    def tstride_at_tstep(self, tstep, mask=True):
        """Returns a Scalar or Pair containing the time interval(s)
        between the start of a given time step and the start of adjacent time
        step(s).

        Input:
            tstep       a Scalar time step index or a Pair of time step
                        indices.
            mask        True to mask values outside the time limits.

        Return:         a Scalar or Pair of strides in seconds.
        """

        if len(self.shape) == 1:
            return (self.time_at_tstep(tstep + 1, mask=mask) -
                    self.time_at_tstep(tstep, mask=mask))
        elif len(self.shape) == 2:
            now = self.time_at_tstep(tstep)
            return Pair.from_scalars(self.time_at_tstep(tstep + (1,0),
                                                        mask=mask) - now,
                                     self.time_at_tstep(tstep + (0,1),
                                                        mask=mask) - now)
        else:
            raise NotImplementedException("tstride_at_tstep() is not " +
                                    "implemented for cadences larger than 2-D")

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Cadence(unittest.TestCase):

    def runTest(self):

        # No tests here - this is just an abstract superclass

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
