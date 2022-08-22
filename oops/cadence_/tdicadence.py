################################################################################
# oops/cadence_/tdicadence.py: TDICadence subclass of class Cadence
################################################################################

from polymath import *
from oops.cadence_.cadence import Cadence

#*****************************************************************************
# TDICadence
#*****************************************************************************
class TDICadence(Cadence):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A Cadence subclass defining the integration intervals of lines in a TDI
    ("Time Delay and Integration") camera. It returns the time range given an
    index in the TDI direction.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    PACKRAT_ARGS = ['lines', 'tstart', 'tdi_texp', 'tdi_stages', 'tdi_sign']

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, lines, tstart, tdi_texp, tdi_stages, tdi_sign=-1):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """Constructor for a TDICadence.

        Input:
            lines       the number of lines in the detector.

            tstart      the start time of the observation in seconds TDB.

            tdi_texp    the interval in seconds from the start of one TDI step
                        to the start of the next.

            tdi_stages  the number of TDI time steps.

            tdi_sign    +1 if pixel DNs are shifted in the positive direction
                        along the 'ut' or 'vt' axis; -1 if DNs are shifted in
                        the negative direction. Default is -1, suitable for
                        JunoCam.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #-----------------------------------
        # Save the input parameters
        #-----------------------------------
        self.lines = lines
        self.tstart = float(tstart)
        self.tdi_texp = float(tdi_texp)
        self.tdi_stages = tdi_stages
        self.tdi_sign = tdi_sign

        self._tdi_upward = (self.tdi_sign > 0)

        #-----------------------------------
        # Fill in the required attributes
        #-----------------------------------
        self.time = (self.tstart, self.tstart + self.tdi_texp * self.tdi_stages)
        self.midtime = 0.5 * (self.time[0] + self.time[1])
        self.lasttime = self.time[1] - self.tdi_texp
        self.shape = (self.lines,)
        self.is_continuous = True
    #===========================================================================



    #===========================================================================
    # time_at_tstep
    #===========================================================================
    def time_at_tstep(self, tstep, mask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the min time(s) associated with the given time step(s).

        This method supports non-integer step values. Note that it overloads the
        standard Cadence.time_at_tstep() method with an additional argument, the
        line number.

        Input:
            tstep       a Scalar time step index or a Pair of indices.
            mask        True to mask values outside the time limits.

        Return:         a Scalar of times in seconds TDB.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        tstep_int = Scalar.as_scalar(tstep).as_int()
        (time_min, time_max) = self.time_range_at_tstep(tstep_int, mask=mask)

        # something is wrong here, as tfrac is not defined...
        time = time_min + tfrac * (time_max - time_min)
        return time
    #===========================================================================



#    #===========================================================================
#    # time_range_at_tstep
#    #===========================================================================
#    def time_range_at_tstep(self, tstep, mask=True):
#        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#        """
#        Return the range of time(s) for the given integer time step(s).
#
#        Input:
#            tstep       a Scalar time step index or a Pair of indices. For this
#                        class
#            mask        True to mask values outside the time limits.
#
#        Return:         (time_min, time_max)
#            time_min    a Scalar defining the minimum time associated with the
#                        index. It is given in seconds TDB.
#            time_max    a Scalar defining the maximum time value.
#        """
#        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#        tstep_int = Scalar.as_scalar(tstep).as_int()
#        tstep_int = tstep_int.clip(0, self.lines, remask=mask)

#        if self._tdi_upward:
#            offset = tstep_int + 1
#        else:
#            offset = Scalar.minimum(self.lines - tstep_int, 1)
#
#        time0 = Scalar.maximum(self.time[0],
#                               self.time[1] - offset * self.tdi_texp)
#        return (time0, self.time[1])
#    #===========================================================================



    #===========================================================================
    # time_range_at_tstep
    #===========================================================================
    def time_range_at_tstep(self, tstep, mask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the range of time(s) for the given integer time step(s).

        Input:
            tstep       a Scalar time step index or a Pair of indices. For this
                        class
            mask        True to mask values outside the time limits.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        tstep_int = Scalar.as_scalar(tstep).as_int()
        tstep_int = tstep_int.clip(0, self.lines, remask=mask)

        if self._tdi_upward:
            offset = Scalar.minimum(tstep_int + 1, self.tdi_stages)
        else:
            offset = Scalar.minimum(self.lines - tstep_int, self.tdi_stages)

        time0 = self.time[1] - offset * self.tdi_texp

        return (time0, self.time[1])
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
        time = Scalar.as_scalar(time)

        return (time >= self.time[0]) & (time <= self.time[1])
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
        return TDICadence(self.tstart + secs, self.tdi_texp, self.tdi_stages,
                          self.tdi_sign, self.lines)
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
        return self
    #===========================================================================

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_TDICadence(unittest.TestCase):

    def runTest(self):

        pass        # Needed!

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

