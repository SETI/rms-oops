################################################################################
# oops/cadence/__init__.py: Abstract class Cadence
################################################################################

from polymath import Scalar, Pair

class Cadence(object):
    """Cadence is an abstract class that defines the timing of an observation.

    At minimum, these attributes are required:
        time            a tuple defining the start time and end time of the
                        observation overall, in seconds TDB.
        midtime         the mid-time of the observation, in seconds TDB.
        lasttime        the start-time of the last time step, in seconds TDB.
        shape           a tuple of integers defining the shape of the indices.
        is_continuous   True if the cadence contains no gaps in time between
                        the start and end.
        is_unique       True if no times inside the cadence are associated with
                        more than one time step.
        min_tstride     minimum time interval between one tstep and the next.
        max_tstride     maximum time interval between one tstep and the next.
    """

    ############################################################################
    # Methods to be defined for each Cadence subclass
    ############################################################################

    def __init__(self):
        """A constructor."""

        pass

    #===========================================================================
    def time_at_tstep(self, tstep, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values.

        Input:
            tstep       a Scalar or Pair of time step index values.
            remask      True to mask values outside the time limits.
            derivs      True to include derivatives of tstep in the returned
                        time.
            inclusive   True to treat the exact maximum size of the cadence as
                        part of the cadence; False to exclude it.

        Return:         a Scalar of times in seconds TDB.
        """

        raise NotImplementedError("time_at_tstep() is not implemented")

    #===========================================================================
    def time_range_at_tstep(self, tstep, remask=False, inclusive=True):
        """The range of times for the given integer time step.

        Input:
            tstep       a Scalar or Pair of time step index values.
            remask      True to mask values outside the time limits.
            inclusive   True to treat the exact maximum size of the cadence as
                        part of the cadence; False to exclude it.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        raise NotImplementedError("time_range_at_tstep() " +
                                  "is not implemented")

    #===========================================================================
    def tstep_at_time(self, time, remask=False, derivs=False):
        """Time step for the given time.

        This method supports non-integer time values and returns non-integer
        time steps.

        Input:
            time        a Scalar of times in seconds TDB.
            remask      True to mask time values not sampled within the cadence.
            derivs      True to include derivatives of time in the returned
                        tstep.

        Return:         a Scalar or Pair of time step indices.
        """

        raise NotImplementedError("tstep_at_time() is not implemented")

    #===========================================================================
    def time_is_outside(self, time, inclusive=True):
        """A Boolean mask of times that fall outside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to treat the end time of an interval as inside;
                        False to treat it as outside. The start time of an
                        interval is always treated as inside.

        Return:         a Boolean array indicating which time values are not
                        sampled by the cadence.
        """

        time = Scalar.as_scalar(time, recursive=False)

        # Default behavior is to treat all times between start and stop as
        # inside. Discontinuous subclasses need to override.

        if inclusive:
            return time.tvl_lt(self.time[0]) | time.tvl_gt(self.time[1])
        else:
            return time.tvl_lt(self.time[0]) | time.tvl_ge(self.time[1])

    #===========================================================================
    def time_shift(self, secs):
        """Construct a duplicate of this Cadence with all times shifted by given
        amount.

        Input:
            secs        the number of seconds to shift the time later.
        """

        raise NotImplementedError("time_shift() is not implemented")

    #===========================================================================
    def as_continuous(self):
        """Construct a shallow copy of this Cadence, forced to be continuous."""

        raise NotImplementedError("as_continuous() is not implemented")

    ############################################################################
    # Methods probably not requiring overrides
    ############################################################################

    def time_is_inside(self, time, inclusive=True):
        """A Boolean mask of times that fall inside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to treat the end time of an interval as inside;
                        False to treat it as outside. The start time of an
                        interval is always treated as inside.

        Return:         a Boolean array indicating which time values are
                        sampled by the cadence.
        """

        return self.time_is_outside(time, inclusive=inclusive).logical_not()

    #===========================================================================
    def tstride_at_tstep(self, tstep, remask=False):
        """The time intervals for the given time steps.

        Input:
            tstep       a Scalar time step index or a Pair of time step
                        indices.
            remask      True to mask values outside the time limits.

        Return:         a Scalar or Pair of strides in seconds.
        """

        if len(self.shape) == 1:
            return (self.time_at_tstep(tstep + 1, remask=remask) -
                    self.time_at_tstep(tstep, remask=remask))
        elif len(self.shape) == 2:
            now = self.time_at_tstep(tstep)
            return Pair.from_scalars(self.time_at_tstep(tstep + (1,0),
                                                        remask=remask) - now,
                                     self.time_at_tstep(tstep + (0,1),
                                                        remask=remask) - now)
        else:
            raise NotImplementedError("tstride_at_tstep() is not " +
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
