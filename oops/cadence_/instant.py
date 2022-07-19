################################################################################
# oops/cadence_/instant.py: Class Instant
################################################################################

from polymath import *
from oops.cadence_.cadence import Cadence

#*******************************************************************************
# Instant
#*******************************************************************************
class Instant(Cadence):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Instant is a class that represents the timing of an observation as an
    Scalar time of arbitrary shape.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    PACKRAT_ARGS = ['tdb', '+is_continuous']

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, tdb):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for an Instant.

        Input:
            tdb         a time Scalar in seconds TDB.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.tdb = Scalar.as_scalar(tdb, recursive=False).as_float()

        self.shape = self.tdb.shape
        self.time = (self.tdb.min(), self.tdb.max())
        self.midtime = 0.5 * (self.time[0] + self.time[1])
        self.lasttime = self.tdb.max()
        self.is_continuous = False
    #===========================================================================



    #===========================================================================
    # time_at_tstep
    #===========================================================================
    def time_at_tstep(self, tstep, mask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the time(s) associated with the given time step(s).
        
        This method supports non-integer step values.

        Input:
            tstep       a Scalar time step index or a Pair of indices.
            mask        True to mask values outside the time limits.

        Return:         a Scalar of times in seconds TDB.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.tdb
    #===========================================================================



    #===========================================================================
    # time_range_at_tstep
    #===========================================================================
    def time_range_at_tstep(self, tstep, mask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the range of time(s) for the given integer time step(s).

        Input:
            indices     a Scalar time step index or a Pair of indices.
            mask        True to mask values outside the time limits.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return (self.tdb, self.tdb)
    #===========================================================================



    #===========================================================================
    # tstep_at_time
    #===========================================================================
    def tstep_at_time(self, time, mask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the time step(s) for given time(s).

        This method supports non-integer time values.

        Input:
            time        a Scalar of times in seconds TDB.
            mask        True to mask time values not sampled within the cadence.

        Return:         a Scalar or Pair of time step indices.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return Scalar(np.zeros(self.shape), self.tdb != time)
    #===========================================================================



    #===========================================================================
    # time_is_inside
    #===========================================================================
    def time_is_inside(self, time, inclusive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return which time(s) fall inside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to include the end moment of a time interval;
                        False to exclude.

        Return:         a Boolean array indicating which time values are
                        sampled by the cadence. A masked time results in a
                        value of False, not a masked Boolean.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return Scalar(time == self.tdb)
    #===========================================================================



    #===========================================================================
    # time_shift
    #===========================================================================
    def time_shift(self, secs):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a duplicate with all times shifted by given amount."

        Input:
            secs        the number of seconds to shift the time later.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return Instant(self.tdb + secs)
    #===========================================================================



    #===========================================================================
    # as_continuous
    #===========================================================================
    def as_continuous(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a shallow copy forced to be continuous.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        instant = Instant(self.tdb)
        instant.is_continuoue = True
        return instant
    #===========================================================================


#*******************************************************************************

################################################################################
# UNIT TESTS
################################################################################

import unittest

#*******************************************************************************
# Test_Instant
#*******************************************************************************
class Test_Instant(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):

        # No tests here - TBD

        pass
    #===========================================================================


#*******************************************************************************



########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
