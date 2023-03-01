################################################################################
# oops/cadence/tdicadence.py: TDICadence subclass of class Cadence
################################################################################

from polymath     import Scalar
from oops.cadence import Cadence

class TDICadence(Cadence):
    """A Cadence subclass defining the integration intervals of lines in a TDI
    ("Time Delay and Integration") camera. The tstep index matches the line
    index in the TDI detector.
    """

    def __init__(self, lines, tstart, tdi_texp, tdi_stages, tdi_sign=-1):
        """Constructor for a TDICadence.

        Input:
            lines       the number of lines in the detector. This corresponds to
                        the number of time steps in the cadence.
            tstart      the start time of the observation in seconds TDB.
            tdi_texp    the interval in seconds from the start of one TDI step
                        to the start of the next.
            tdi_stages  the number of TDI time steps, 1 to number of lines.
            tdi_sign    +1 if pixel DNs are shifted in the positive direction
                        along the 'ut' or 'vt' axis; -1 if DNs are shifted in
                        the negative direction. Default is -1, suitable for
                        JunoCam.
        """

        # Save the input parameters
        self.lines = int(lines)
        self.tstart = float(tstart)
        self.tdi_texp = float(tdi_texp)
        self.tdi_stages = int(tdi_stages)
        self.tdi_sign = 1 if tdi_sign > 0 else -1

        if self.tdi_stages < 1 or self.tdi_stages > self.lines:
            raise ValueError('invalid TDICadence inputs: ' +
                             'lines=%d; tdi_stages=%d' % (lines, tdi_stages))

        self._tdi_upward = (self.tdi_sign > 0)
        self._max_shifts = self.tdi_stages - 1
        self._max_line = self.lines - 1

        # Number of lines that are always active
        self._perm_lines = self.lines - self._max_shifts

        # Fill in the required attributes
        self.time = (self.tstart, self.tstart + self.tdi_texp * self.tdi_stages)
        self.midtime = 0.5 * (self.time[0] + self.time[1])
        self.lasttime = self.time[-1] - self.tdi_texp
        self.shape = (self.lines,)
        self.is_continuous = True
        self.is_unique = (self.tdi_stages == 1)
        self.min_tstride = 0.
        self.max_tstride = tdi_texp

        self._scalar_end_time = Scalar(self.time[1])

    def __getstate__(self):
        return (self.lines, self.tstart, self.tdi_texp, self.tdi_stages,
                self.tdi_sign)

    def __setstate__(self, state):
        self.__init__(*state)

    ############################################################################
    # Methods unique to this class
    ############################################################################

    def tdi_shifts_at_line(self, line, remask=False, inclusive=True):
        """The number of TDI shifts at the given image line (or tstep).

        Input:
            line        a Scalar line number.
            remask      True to mask values outside the time limits.
            inclusive   True to treat the end time of the cadence as inside the
                        cadence. If inclusive is False and remask is True, the
                        end time will be masked.

        Return:         an integer Scalar defining the number of TDI shifts at
                        this line number.
        """

        line = Scalar.as_scalar(line, recursive=False)
        line = line.int(top=self.lines, remask=remask, inclusive=inclusive)

        if self._tdi_upward:
            shifts = line
        else:
            shifts = self._max_line - line

        return shifts.clip(0, self._max_shifts, remask=False)

    #===========================================================================
    def tdi_shifts_after_time(self, time, remask=False, inclusive=True):
        """The number of TDI shifts at the given time.

        Input:
            time        Scalar of optional absolute time in seconds.
            remask      True to mask values outside the time limits.
            inclusive   True to treat the end time of the cadence as inside the
                        cadence. If inclusive is False and remask is True, the
                        end time will be masked.

        Return:         an integer Scalar defining the number of TDI shifts that
                        will occur after this time in the exposure.
        """

        time = Scalar.as_scalar(time, recursive=False)
        tstep = (time - self.time[0]) / self.tdi_texp
        tstep_int = tstep.int(top=self.tdi_stages,
                              remask=remask, inclusive=inclusive)
        return (self._max_shifts - tstep_int).clip(0, self.tdi_stages,
                                                      remask=remask)

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
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Scalar of times in seconds TDB.
        """

        tstep = Scalar.as_scalar(tstep, recursive=derivs)
        tstep_int = tstep.int(top=self.lines, remask=remask,
                              inclusive=inclusive, clip=True)
        tstep_frac = (tstep - tstep_int).clip(0, 1, inclusive=inclusive,
                                                    remask=False)

        (time_min,
         time_max) = self.time_range_at_tstep(tstep_int, remask=False)

        return time_min + tstep_frac * (time_max - time_min)

    #===========================================================================
    def time_range_at_tstep(self, tstep, remask=False, inclusive=True):
        """The range of times for the given time step.

        Input:
            tstep       a Scalar of time step index values.
            remask      True to mask values outside the time limits.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        stages = self.tdi_shifts_at_line(tstep, remask=remask,
                                                inclusive=inclusive) + 1

        time0 = self.time[1] - stages * self.tdi_texp
        time1 = Scalar.filled(time0.shape, self.time[1], mask=time0.mask)
        return (time0, time1)

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

        Return:         a Scalar of time step index values.
        """

        if self.tdi_stages > 1:
            raise NotImplementedError('TDICadence.tstep_at_time cannot be ' +
                                      'implemented; time values are not unique')

        time = Scalar.as_scalar(time, recursive=derivs)
        tstep = (time - self.time[0]) / self.tdi_texp
        return tstep.clip(0, 1, inclusive=inclusive, remask=remask)

    #===========================================================================
    def tstep_range_at_time(self, time, remask=False, inclusive=True):
        """Integer range of time steps active at the given time.

        Input:
            time        a Scalar of times in seconds TDB.
            remask      True to mask time values not sampled within the cadence.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         (tstep_min, tstep_max)
            tstep_min   minimum Scalar time step containing the given time.
            tstep_max   minimum Scalar time step after the given time.

        Returned tstep_min will always be in the allowed range for the cadence,
        inclusive, regardless of masking. If the time is not inside the cadence,
        tstep_max == tstep_min.
        """

        time = Scalar.as_scalar(time, recursive=False)
        shifts = (time - self.time[0]) / self.tdi_texp

        # remask = True here; fix it below
        shifts = shifts.int(top=self.tdi_stages, remask=True,
                            inclusive=inclusive, clip=True)

        if self._tdi_upward:
            line_min = self._max_shifts - shifts
            line_max = Scalar.filled(shifts.shape, self.lines)
            line_max[shifts.mask] = line_min[shifts.mask]
        else:
            line_min = Scalar.zeros(shifts.shape, dtype='int', mask=shifts.mask)
            line_max = self._perm_lines + shifts
            line_max[shifts.mask] = line_min[shifts.mask]

        if remask:
            line_min = line_min.remask(shifts.mask)
            line_max = line_max.remask(shifts.mask)
        else:
            line_min = line_min.remask(time.mask)
            line_max = line_max.remask(time.mask)

        return (line_min, line_max)

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
import numpy as np

class Test_TDICadence(unittest.TestCase):

    def runTest(self):

        ########################################
        # 10 lines, 2 stages, TDI downward, 100-120
        ########################################

        cad = TDICadence(10, 100., 10., 2)
        case_tdicadence_10_100_10_2_down(self, cad)

        ########################################
        # 10 lines, 2 stages, TDI upward
        ########################################

        cad = TDICadence(10, 100., 10., 2, tdi_sign=1)
        case_tdicadence_10_100_10_2_up(self, cad)

        ########################################
        # 100 lines, 100 stages, TDI downward
        ########################################

        cad = TDICadence(100, 1000., 10., 100)
        case_tdicadence_100_1000_10_100_down(self, cad)

        ########################################
        # 10 lines, one stage
        ########################################

        cad = TDICadence(10, 100., 10., 1)
#         print(cad.time_at_tstep(10))
        case_tdicadence_10_100_10_1(self, cad)

def case_tdicadence_10_100_10_2_down(self, cad):

    # time_range_at_tstep
    self.assertEqual(cad.time_range_at_tstep(-1), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(-1, remask=True), (Scalar.MASKED, Scalar.MASKED))
    self.assertEqual(cad.time_range_at_tstep(0), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(8), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(9), (110., 120.))
    self.assertEqual(cad.time_range_at_tstep(9.5),(110., 120.))
    self.assertEqual(cad.time_range_at_tstep(10), (110., 120.))
    self.assertEqual(cad.time_range_at_tstep(10, inclusive=False,
                                                 remask=False), (110., 120.))
    self.assertEqual(cad.time_range_at_tstep(10, inclusive=False,
                                                 remask=True), (Scalar.MASKED, Scalar.MASKED))
    self.assertEqual(cad.time_range_at_tstep(11), (110., 120.))
    self.assertEqual(cad.time_range_at_tstep(11, inclusive=False,
                                                 remask=False), (110., 120.))
    self.assertEqual(cad.time_range_at_tstep(11, inclusive=False,
                                                 remask=True), (Scalar.MASKED, Scalar.MASKED))

    tstep = Scalar(([0,1],[2,9]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[0].vals ==
                           [[100,100],[100,110]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[0].mask == tstep.mask))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[1].mask == tstep.mask))

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[0].vals ==
                           [[100,100],[100,110]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[0].mask == tstep.mask))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[1].mask == tstep.mask))

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[0].vals ==
                           [[100,100],[100,110]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[0].mask == tstep.mask))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[1].mask == tstep.mask))

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[0].vals == [[100,100],[100,110]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[0].mask == [[0,0],[1,1]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[1].mask == [[0,0],[1,1]]))

    # time_at_tstep
    self.assertEqual(cad.time_at_tstep(-1.), 100.)
    self.assertEqual(cad.time_at_tstep(-1., remask=True), Scalar.MASKED)
    self.assertEqual(cad.time_at_tstep(0. ), 100.)
    self.assertEqual(cad.time_at_tstep(0.5), 110.)
    self.assertEqual(cad.time_at_tstep(0.9), 118.)
    self.assertEqual(cad.time_at_tstep(1. ), 100.)
    self.assertEqual(cad.time_at_tstep(1.5), 110.)
    self.assertEqual(cad.time_at_tstep(1.9), 118.)
    self.assertEqual(cad.time_at_tstep(9. ), 110.)
    self.assertEqual(cad.time_at_tstep(9.5), 115.)
    self.assertEqual(cad.time_at_tstep(10.), 120.)
    self.assertEqual(cad.time_at_tstep(10., remask=True,
                                            inclusive=True), 120.)
    self.assertEqual(cad.time_at_tstep(10., remask=True,
                                            inclusive=False), Scalar.MASKED)

    tstep = Scalar(([0,1],[9,10]),([False,True],[False,False]))
    self.assertTrue(np.all(cad.time_at_tstep(tstep).vals == [[100,100],[110,120]]))
    self.assertTrue(np.all(cad.time_at_tstep(tstep).mask == tstep.mask))
    self.assertTrue(np.all(cad.time_at_tstep(tstep, remask=True,
                                             inclusive=False).mask == [[0,1],[0,1]]))

    # time_at_tstep, derivs
    tstep = Scalar([-1, 0, 0.5, 0.9, 1.,1.5, 1.9, 9, 9.5, 10])
    tstep.insert_deriv('t', Scalar(np.arange(10.)))
    self.assertEqual(cad.time_at_tstep(tstep),
                    [100,100,110,118,100,110,118,110,115,120])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True),
                    [100,100,110,118,100,110,118,110,115,120])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True),
                    [Scalar.MASKED,100,110,118,100,110,118,110,115,120])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True, inclusive=False),
                    [Scalar.MASKED,100,110,118,100,110,118,110,115,Scalar.MASKED])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True).d_dt,
                    [0,20,40,60,80,100,120,70,80,90])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True).d_dt,
                    [Scalar.MASKED,20,40,60,80,100,120,70,80,90])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True, inclusive=False).d_dt,
                    [Scalar.MASKED,20,40,60,80,100,120,70,80,Scalar.MASKED])

    # tstep_range_at_time
    self.assertEqual(cad.tstep_range_at_time(100.), (0, 9))
    self.assertEqual(cad.tstep_range_at_time(109.), (0, 9))
    self.assertEqual(cad.tstep_range_at_time(110.), (0, 10))
    self.assertEqual(cad.tstep_range_at_time(120.), (0, 10))

    # self.assertEqual(cad.tstep_range_at_time(120., inclusive=False), (0, 0))
    (test0, test1) = cad.tstep_range_at_time(120., inclusive=False)
    self.assertEqual(test0, test1)
    self.assertEqual(cad.tstep_range_at_time(120., remask=True,
                                                   inclusive=True), (0, 10))
    self.assertEqual(cad.tstep_range_at_time(120., remask=True,
                                                   inclusive=False), (Scalar.MASKED, Scalar.MASKED))

    time = Scalar([100,110,120],[False,False,False])
    self.assertTrue(np.all(cad.tstep_range_at_time(time)[0].vals == (0,0,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time)[1].vals == (9,10,10)))

    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[0].vals == (0,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[1].vals == (9,10,0)))
    (test0, test1) = cad.tstep_range_at_time(time, inclusive=False)
    self.assertTrue(np.all(test0.vals[:2] == (0,0)))
    self.assertTrue(np.all(test1.vals[:2] == (9,10)))
    self.assertEqual(test0.vals[2], test1.vals[2])  # zero range is required, specific values are not

    self.assertTrue(not np.any(cad.tstep_range_at_time(time, inclusive=False)[0].mask))
    self.assertTrue(not np.any(cad.tstep_range_at_time(time, inclusive=False)[1].mask))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals == (0,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals == (9,10,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals == test0.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals == test1.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].mask == (0,0,1)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].mask == (0,0,1)))

    time = Scalar([100,110,120],[True,False,False])
    # self.assertTrue(np.all(cad.tstep_range_at_time(time)[0].vals == (0,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time)[1].vals == (0,10,10)))
    (test0, test1) = cad.tstep_range_at_time(time)
    self.assertEqual(test0.vals[0], test1.vals[0])  # zero range is required, specific values are not
    self.assertTrue(np.all(test0.vals[1:] == (0,0)))
    self.assertTrue(np.all(test1.vals[1:] == (10,10)))

    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[0].vals == (0,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[1].vals == (0,10,0)))
    (test0, test1) = cad.tstep_range_at_time(time, inclusive=False)
    self.assertEqual(test0.vals[0], test1.vals[0])  # zero range is required, specific values are not
    self.assertEqual(test0.vals[1], 0)
    self.assertEqual(test1.vals[1], 10)
    self.assertEqual(test0.vals[2], test1.vals[2])  # zero range is required, specific values are not

    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[0].mask == (1,0,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[1].mask == (1,0,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals == test0.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals == test1.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].mask == (1,0,1)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].mask == (1,0,1)))

    # tstride_at_tstep
    self.assertEqual(cad.tstride_at_tstep(0), 0)
    self.assertEqual(cad.tstride_at_tstep(8), 10)
    self.assertEqual(cad.tstride_at_tstep(8, sign=-1), 0)
    self.assertEqual(cad.tstride_at_tstep(9), 10)
    self.assertEqual(cad.tstride_at_tstep(9, sign=-1), 10)
    self.assertEqual(cad.tstride_at_tstep(10), 10)

def case_tdicadence_10_100_10_2_up(self, cad):

    # time_range_at_tstep
    self.assertEqual(cad.time_range_at_tstep(-1), (110., 120.))
    self.assertEqual(cad.time_range_at_tstep(-1, remask=True), (Scalar.MASKED, Scalar.MASKED))
    self.assertEqual(cad.time_range_at_tstep(0), (110., 120.))
    self.assertEqual(cad.time_range_at_tstep(8), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(9), (100., 120.))

    self.assertEqual(cad.time_range_at_tstep(10), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(10, inclusive=False,
                                                 remask=False), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(10, inclusive=False,
                                                 remask=True), (Scalar.MASKED, Scalar.MASKED))
    self.assertEqual(cad.time_range_at_tstep(11), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(11, inclusive=False,
                                                 remask=False), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(11, inclusive=False,
                                                 remask=True), (Scalar.MASKED, Scalar.MASKED))

    tstep = Scalar(([0,1],[2,9]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[0].vals ==
                           [[110,100],[100,100]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[0].mask == tstep.mask))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[1].mask == tstep.mask))

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[0].vals ==
                           [[110,100],[100,100]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[0].mask == tstep.mask))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[1].mask == tstep.mask))

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[0].vals ==
                           [[110,100],[100,100]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[0].mask == tstep.mask))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[1].mask == tstep.mask))

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[0].vals == [[110,100],[100,100]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[0].mask == [[0,0],[1,1]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[1].mask == [[0,0],[1,1]]))

    # time_at_tstep
    self.assertEqual(cad.time_at_tstep(-1.), 110.)
    self.assertEqual(cad.time_at_tstep(-1., remask=True), Scalar.MASKED)
    self.assertEqual(cad.time_at_tstep(0. ), 110.)
    self.assertEqual(cad.time_at_tstep(0.5), 115.)
    self.assertEqual(cad.time_at_tstep(0.9), 119.)
    self.assertEqual(cad.time_at_tstep(1. ), 100.)
    self.assertEqual(cad.time_at_tstep(1.5), 110.)
    self.assertEqual(cad.time_at_tstep(1.9), 118.)
    self.assertEqual(cad.time_at_tstep(9. ), 100.)
    self.assertEqual(cad.time_at_tstep(9.5), 110.)
    self.assertEqual(cad.time_at_tstep(10.), 120.)
    self.assertEqual(cad.time_at_tstep(10., remask=True,
                                            inclusive=True), 120.)
    self.assertEqual(cad.time_at_tstep(10., remask=True,
                                            inclusive=False), Scalar.MASKED)

    self.assertEqual(cad.tstep_range_at_time(100.), (1, 10))
    self.assertEqual(cad.tstep_range_at_time(109.), (1, 10))
    self.assertEqual(cad.tstep_range_at_time(110.), (0, 10))
    self.assertEqual(cad.tstep_range_at_time(120.), (0, 10))

    # self.assertEqual(cad.tstep_range_at_time(120., inclusive=False), (0, 0))
    (test0, test1) = cad.tstep_range_at_time(120., inclusive=False)
    self.assertEqual(test0, test1)  # zero range is required; specific values are not

    self.assertEqual(cad.tstep_range_at_time(120., remask=True,
                                                   inclusive=True), (0, 10))
    self.assertEqual(cad.tstep_range_at_time(120., remask=True,
                                                   inclusive=False), (Scalar.MASKED, Scalar.MASKED))

    # time_at_tstep, derivs
    tstep = Scalar([-1, 0, 0.5, 0.9, 1.,1.5, 1.9, 9, 9.5, 10])
    tstep.insert_deriv('t', Scalar(np.arange(10.)))
    self.assertEqual(cad.time_at_tstep(tstep),
                    [110,110,115,119,100,110,118,100,110,120])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True),
                    [110,110,115,119,100,110,118,100,110,120])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True),
                    [Scalar.MASKED,110,115,119,100,110,118,100,110,120])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True, inclusive=False),
                    [Scalar.MASKED,110,115,119,100,110,118,100,110,Scalar.MASKED])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True).d_dt,
                    [0,10,20,30,80,100,120,140,160,180])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True).d_dt,
                    [Scalar.MASKED,10,20,30,80,100,120,140,160,180])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True, inclusive=False).d_dt,
                    [Scalar.MASKED,10,20,30,80,100,120,140,160,Scalar.MASKED])

    # tstep_range_at_time
    self.assertEqual(cad.tstep_range_at_time(100.), (1, 10))
    self.assertEqual(cad.tstep_range_at_time(109.), (1, 10))
    self.assertEqual(cad.tstep_range_at_time(110.), (0, 10))
    self.assertEqual(cad.tstep_range_at_time(120.), (0, 10))

    # self.assertEqual(cad.tstep_range_at_time(120., inclusive=False), (0, 0))
    (test0, test1) = cad.tstep_range_at_time(120., inclusive=False)
    self.assertEqual(test0, test1)  # zero range is required; specific values are not

    self.assertEqual(cad.tstep_range_at_time(120., remask=True,
                                                   inclusive=True), (0, 10))
    self.assertEqual(cad.tstep_range_at_time(120., remask=True,
                                                   inclusive=False), (Scalar.MASKED, Scalar.MASKED))

    time = Scalar([100,110,120],[False,False,False])
    self.assertTrue(np.all(cad.tstep_range_at_time(time)[0].vals == (1,0,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time)[1].vals == (10,10,10)))

    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[0].vals == (1,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[1].vals == (10,10,0)))
    (test0, test1) = cad.tstep_range_at_time(time, inclusive=False)
    self.assertTrue(np.all(test0.vals[:2] == (1,0)))
    self.assertTrue(np.all(test1.vals[:2] == (10,10)))
    self.assertEqual(test0.vals[2], test1.vals[2])  # zero range is required, specific values are not

    self.assertTrue(not np.any(cad.tstep_range_at_time(time, inclusive=False)[0].mask))
    self.assertTrue(not np.any(cad.tstep_range_at_time(time, inclusive=False)[1].mask))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals == (1,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals == (10,10,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals == test0.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals == test1.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].mask == (0,0,1)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].mask == (0,0,1)))

    time = Scalar([100,110,120],[False,True,False])
    # self.assertTrue(np.all(cad.tstep_range_at_time(time)[0].vals == (1,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time)[1].vals == (10,0,10)))
    (test0, test1) = cad.tstep_range_at_time(time)
    self.assertEqual(test0.vals[1], test1.vals[1])  # zero range is required, specific values are not
    self.assertTrue(np.all(test0.vals[0::2] == (1,0)))
    self.assertTrue(np.all(test1.vals[0::2] == (10,10)))

    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[0].vals == (1,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[1].vals == (10,0,0)))
    (test0, test1) = cad.tstep_range_at_time(time, inclusive=False)
    self.assertEqual(test0.vals[0], 1)
    self.assertEqual(test1.vals[0], 10)
    self.assertEqual(test0.vals[1], test1.vals[1])  # zero range is required, specific values are not
    self.assertEqual(test0.vals[2], test1.vals[2])

    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[0].mask == (0,1,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[1].mask == (0,1,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals == (1,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals == (10,0,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals == test0.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals == test1.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].mask == (0,1,1)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].mask == (0,1,1)))

def case_tdicadence_100_1000_10_100_down(self, cad):

    tstep = Scalar(np.arange(100))
    (time0, time1) = cad.time_range_at_tstep(tstep)
    self.assertEqual(time1, 2000.)
    self.assertEqual(time0, 1000. + 10. * tstep)

    self.assertEqual(cad.tstep_range_at_time(1000.), (0, 1))
    self.assertEqual(cad.tstep_range_at_time(1010.), (0, 2))
    self.assertEqual(cad.tstep_range_at_time(1990.), (0, 100))
    # self.assertEqual(cad.tstep_range_at_time(2000., inclusive=False), (0, 0))
    (test0, test1) = cad.tstep_range_at_time(2000., inclusive=False)
    self.assertEqual(test0, test1)
    self.assertEqual(cad.tstep_range_at_time(2000., remask=True,
                                                    inclusive=True), (0, 100))
    self.assertEqual(cad.tstep_range_at_time(2000., remask=True,
                                                    inclusive=False), (Scalar.MASKED, Scalar.MASKED))

    # time_is_inside()
    self.assertEqual(cad.time_is_inside([1000,2000], inclusive=True ), [1,1])
    self.assertEqual(cad.time_is_inside([1000,2000], inclusive=False), [1,0])

def case_tdicadence_10_100_10_1(self, cad):

    self.assertTrue(cad.is_continuous)
    self.assertTrue(cad.is_unique)

    # time_at_tstep()
    self.assertEqual(cad.time_at_tstep(-0.1), 100.)
    self.assertEqual(cad.time_at_tstep(-0.1, remask=False), 100.)
    self.assertEqual(cad.time_at_tstep(-0.1, remask=True ), Scalar.MASKED)
    self.assertEqual(cad.time_at_tstep( 0  ), 100.)
    self.assertEqual(cad.time_at_tstep( 9.5), 105.)
    self.assertEqual(cad.time_at_tstep(10, remask=False), 110.)
    self.assertEqual(cad.time_at_tstep(10, remask=True ), 110.)
    self.assertEqual(cad.time_at_tstep(10, remask=True, inclusive=False), Scalar.MASKED)

    # time_at_tstep(), derivs
    tstep = Scalar((0., 0.5, 10., 20.))
    tstep.insert_deriv('t', Scalar((2,3,4,5)))

    self.assertEqual(cad.time_at_tstep(tstep, remask=False), (100,105,110,110))
    self.assertEqual(cad.time_at_tstep(tstep, remask=True), (100,105,110,Scalar.MASKED))
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True).d_dt, (20,30,40,0))
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, inclusive=False).d_dt, (20,30,0,0))

    # time_is_inside()
    time = ([99,100],[120,140],[145,150])
    self.assertFalse(cad.time_is_inside(90))
    self.assertTrue (cad.time_is_inside(100))
    self.assertTrue (cad.time_is_inside(110))
    self.assertFalse(cad.time_is_inside(110, inclusive=False))
    self.assertFalse(cad.time_is_inside(111))

    # tstep_at_time()
    self.assertEqual(cad.tstep_at_time( 99), 0.)
    self.assertEqual(cad.tstep_at_time( 99, remask=True), Scalar.MASKED)
    self.assertEqual(cad.tstep_at_time(100), 0.)
    self.assertEqual(cad.tstep_at_time(105), 0.5)
    self.assertEqual(cad.tstep_at_time(110), 1.)
    self.assertEqual(cad.tstep_at_time(110, remask=True), 1.)
    self.assertEqual(cad.tstep_at_time(110, remask=True, inclusive=False), Scalar.MASKED)
    self.assertEqual(cad.tstep_at_time(111), 1.)
    self.assertEqual(cad.tstep_at_time(111, remask=True), Scalar.MASKED)

    # tstep_at_time(), derivs
    time = Scalar((90,100,110,140), derivs={'t': Scalar((100,200,300,400))})
    self.assertEqual(cad.tstep_at_time(time, remask=False, derivs=True).d_dt, (0, 20, 30, 0))
    self.assertEqual(cad.tstep_at_time(time, remask=False, derivs=True,
                                                 inclusive=False).d_dt, (0, 20, 0, 0))

    # tstep_range_at_time()
    MASKED_TUPLE = (Scalar.MASKED, Scalar.MASKED)
    self.assertEqual(cad.tstep_range_at_time( 99.), (0,0))
    self.assertEqual(cad.tstep_range_at_time( 99., remask=True), MASKED_TUPLE)
    self.assertEqual(cad.tstep_range_at_time(100.), (0,10))
    self.assertEqual(cad.tstep_range_at_time(105.), (0,10))
    self.assertEqual(cad.tstep_range_at_time(110.), (0,10))
    self.assertEqual(cad.tstep_range_at_time(110., remask=True), (0,10))
    self.assertEqual(cad.tstep_range_at_time(110., remask=True, inclusive=False), MASKED_TUPLE)
    self.assertEqual(cad.tstep_range_at_time(135., remask=True), MASKED_TUPLE)

    tstep0, tstep1 = cad.tstep_range_at_time(110., inclusive=False)
    self.assertEqual(tstep0, tstep1)    # indicates zero range

    tstep0, tstep1 = cad.tstep_range_at_time(135.)
    self.assertEqual(tstep0, tstep1)

    # time_range_at_tstep()
    tstep = Scalar((-1,0,0.5,10,12))
    self.assertEqual(cad.time_range_at_tstep(tstep)[0], 5*[100])
    self.assertEqual(cad.time_range_at_tstep(tstep)[1], 5*[110])

    self.assertEqual(cad.time_range_at_tstep(tstep[0], remask=True), MASKED_TUPLE)
    self.assertEqual(cad.time_range_at_tstep(tstep[1:4], remask=True)[0], 3*[100])
    self.assertEqual(cad.time_range_at_tstep(tstep[1:4], remask=True)[1], 3*[110])
    self.assertEqual(cad.time_range_at_tstep(tstep[4], remask=True), MASKED_TUPLE)

    self.assertEqual(cad.time_range_at_tstep(tstep[1:3], remask=True, inclusive=False)[0], 2*[100])
    self.assertEqual(cad.time_range_at_tstep(tstep[1:3], remask=True, inclusive=False)[1], 2*[110])

    self.assertEqual(cad.time_range_at_tstep(tstep[3], remask=True, inclusive=False), MASKED_TUPLE)

    # tstride_at_tstep
    self.assertEqual(cad.tstride_at_tstep(0), 0)
    self.assertEqual(cad.tstride_at_tstep(0.5), 0)
    self.assertEqual(cad.tstride_at_tstep(9), 10)
    self.assertEqual(cad.tstride_at_tstep(9, sign=-1), 0)
    self.assertEqual(cad.tstride_at_tstep(10), 10)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
