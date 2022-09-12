################################################################################
# oops/cadence/tdicadence.py: TDICadence subclass of class Cadence
################################################################################

from polymath import Qube, Boolean, Scalar, Pair, Vector

from . import Cadence

class TDICadence(Cadence):
    """A Cadence subclass defining the integration intervals of lines in a TDI
    ("Time Delay and Integration") camera.
    """

    def __init__(self, lines, tstart, tdi_texp, tdi_stages, tdi_sign=-1):
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

        # Save the input parameters
        self.lines = lines
        self.tstart = float(tstart)
        self.tdi_texp = float(tdi_texp)
        self.tdi_stages = tdi_stages
        self.tdi_sign = 1 if tdi_sign > 0 else -1

        self._tdi_upward = (self.tdi_sign > 0)
        self._max_shifts = self.tdi_stages - 1
        self._max_line = self.lines - 1

        # Fill in the required attributes
        self.time = (self.tstart, self.tstart + self.tdi_texp * self.tdi_stages)
        self.midtime = 0.5 * (self.time[0] + self.time[1])
        self.lasttime = None            # has no meaningful definition
        self.shape = (self.lines,)
        self.is_continuous = True

    def __getstate__(self):
        return (self.lines, self.tstart, self.tdi_texp, self.tdi_stages,
                self.tdi_sign)

    def __setstate__(self, state):
        self.__init__(*state)

    ############################################################################
    # Methods unique to this class
    ############################################################################

    def tdi_shifts_at_line(self, line, remask=False):
        """The number of TDI shifts at the given image line.

        Note that this method is unique to the TDICadence class.

        Input:
            line        a Scalar line number.
            remask      True to mask values outside the time limits.

        Return:         an integer Scalar defining the number of TDI shifts at
                        this line number
        """

        line = Scalar.as_scalar(line, recursive=False).as_int()

        if self._tdi_upward:
            shifts = line
        else:
            shifts = self._max_line - line

        shifts = shifts.clip(0, self._max_shifts, remask=remask)

    ############################################################################
    # Standard Cadence methods
    ############################################################################

    def time_at_tstep(self, tstep, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values.

        Input:
            tstep       a Scalar of time step index values.
            remask      True to mask values outside the time limits.
            derivs      True to include derivatives of tstep in the returned
                        time.
            inclusive   True to treat the maximum index of the cadence as inside
                        the cadence; False to treat it as outside.

        Return:         a Scalar of times in seconds TDB.
        """

        top = self.tdi_stages if inclusive else None

        tstep_int = Scalar.as_scalar(tstep).int(top=top, remask=remask)
        (time_min,
         time_max) = self.time_range_at_tstep(tstep_int, remask=remask)

        tfrac = tstep - tstep_int
        time = time_min + tfrac * (time_max - time_min)
        return time

    #===========================================================================
    def time_range_at_tstep(self, tstep, remask=False, inclusive=True,
                                         shift=True):
        """The range of times for the given integer time step.

        Input:
            tstep       a Pair of time step index values.
            remask      True to mask values outside the time limits.
            inclusive   True to treat the maximum index of the cadence as inside
                        the cadence; False to treat it as outside.
            shift       True to identify the end moment of the cadence as being
                        part of the last time step.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        stages = self.tdi_shifts_at_line(tstep, remask=remask) + 1

        return (self.time[1] - stages * self.tdi_texp, self.time[1])

    #===========================================================================
    def tstep_at_time(self, time, remask=False, derivs=False, inclusive=True):
        """Time step for the given time.

        This method returns non-integer time steps.

        Input:
            time        a Scalar of times in seconds TDB.
            remask      True to mask time values not sampled within the cadence.
            derivs      True to include derivatives of tstep in the returned
                        time.
            inclusive   True to treat the end time of an interval as inside the
                        cadence; False to treat it as outside. The start time of
                        an interval is always treated as inside.

        Return:         a Pair of time step index values.
        """

        #### TBD

    #===========================================================================
    def time_is_outside(self, time, inclusive=True):
        """A Boolean mask of time(s) that fall outside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to treat the end time of an interval as inside;
                        False to treat it as outside. The start time of an
                        interval is always treated as inside.

        Return:         a Boolean array indicating which time values are not
                        sampled by the cadence.
        """

        return Cadence.time_is_outside(self, time, inclusive)

    #===========================================================================
    def time_shift(self, secs):
        """Construct a duplicate of this Cadence with all times shifted by given
        amount.

        Input:
            secs        the number of seconds to shift the time later.
        """

        return TDICadence(self.tstart + secs, self.tdi_texp, self.tdi_stages,
                          self.tdi_sign, self.lines)

    #===========================================================================
    def as_continuous(self):
        """A shallow copy of this cadence, forced to be continuous."""

        return self

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

