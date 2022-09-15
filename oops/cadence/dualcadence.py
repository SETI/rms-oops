################################################################################
# oops/cadence/dualcadence.py: DualCadence subclass of class Cadence
################################################################################

from polymath   import Qube, Boolean, Scalar, Pair, Vector

from .          import Cadence
from .metronome import Metronome

class DualCadence(Cadence):
    """A Cadence subclass in which time steps are defined by a pair of cadences.
    """

    #===========================================================================
    def __init__(self, long, short):
        """Constructor for a DualCadence.

        Input:
            long        the long or outer cadence. It defines the larger steps
                        of the cadence, including the overall start time.
            short       the short or inner cadence. It defines the time steps
                        that break up the outer cadence, including the exposure
                        time.
        """

        self.long = long.as_continuous()
        self.short = short
        self.shape = self.long.shape + self.short.shape
        assert len(self.long.shape) == 1
        assert len(self.short.shape) == 1

        self._short_time0 = self.short.time[0]
        self._short_duration = self.short.time[1] - self._short_time0
        assert self._short_duration <= self.long.min_tstride    # no time overlaps

        self.time = (self.long.time[0],
                     self.long.lasttime + self._short_duration)
        self.midtime = (self.time[0] + self.time[1]) * 0.5
        self.lasttime = self.long.lasttime + (self.short.lasttime -
                                              self._short_time0)

        self.is_continuous = (self.short.is_continuous and
                              self._short_duration >= self.long.max_tstride)
                            # Note that self.long is already continuous

        self.is_unique = (self.long.is_unique and
                          self.short.is_unique and
                          self._short_duration <= self.long.min_tstride)

        self.min_tstride = self.short.min_tstride
        self.max_tstride = max(self.long.max_tstride - self._short_duration,
                               self.short.max_tstride)

        self._max_long_tstep = self.long.shape[0] - 1

    def __getstate__(self):
        return (self.long, self.short)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def time_at_tstep(self, tstep, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values.

        Input:
            tstep       a Pair of time step index values.
            remask      True to mask values outside the time limits.
            derivs      True to include derivatives of tstep in the returned
                        time.
            inclusive   True to treat the maximum index of the cadence as inside
                        the cadence; False to treat it as outside.

        Return:         a Scalar of times in seconds TDB.
        """

        tstep = Pair.as_pair(tstep, recursive=derivs)
        (long_tstep, short_tstep) = tstep.to_scalars()

        long_int = long_tstep.int(top=self.shape[0],
                                  remask=remask,
                                  inclusive=inclusive,
                                  shift=inclusive)

        long_time = self.long.time_at_tstep(long_int,
                                            remask=False,
                                            derivs=False,
                                            inclusive=inclusive)

        short_time = self.short.time_at_tstep(short_tstep,
                                              remask=remask,
                                              derivs=derivs,
                                              inclusive=inclusive)

        time = long_time + short_time - self._short_time0

        # The extreme corner requires special handling. If inclusive == True,
        # this must get set to the end time overall.
        if inclusive:
            mask = ((long_tstep.vals == self.shape[0]) &
                    (short_tstep.vals == self.shape[1]))
            if np.any(mask):
                time[mask] = self.time[1]

        return time

    #===========================================================================
    def time_range_at_tstep(self, tstep, remask=False, inclusive=True,
                                         shift=True):
        """The range of times for the given time step.

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

        tstep = Pair.as_pair(tstep, recursive=False)
        (long_tstep, short_tstep) = tstep.to_scalars()

        long_int = long_tstep.int(top=self.shape[0],
                                  remask=remask,
                                  inclusive=inclusive,
                                  shift=shift)

        long_time0 = self.long.time_range_at_tstep(long_int,
                                                   remask=remask,
                                                   inclusive=inclusive,
                                                   shift=False)[0]

        short_times = self.short.time_range_at_tstep(short_tstep,
                                                     remask=remask,
                                                     inclusive=inclusive,
                                                     shift=shift)

        long_ref = long_time0 - self._short_time0
        return (long_ref + short_times[0], long_ref + short_times[1])

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

        tstep0 = self.long.tstep_at_time(time,
                                         remask=remask,
                                         derivs=False,
                                         inclusive=False)

        tstep0_int = tstep0.int(top=self.shape[0],
                                remask=False,
                                inclusive=False)

        time0 = self.long.time_at_tstep(tstep0_int,
                                        remask=remask,
                                        inclusive=False)

        tstep1 = self.short.tstep_at_time(self._short_time0 + time - time0,
                                          remask=remask,
                                          derivs=derivs,
                                          inclusive=inclusive)

        return Pair.from_scalars(tstep0_int, tstep1)

    #===========================================================================
    def time_is_outside(self, time, inclusive=True):
        """A Boolean mask of times that fall outside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to treat the end time of an interval as inside the
                        cadence; False to treat it as outside. The start time of
                        an interval is always treated as inside.

        Return:         a Boolean array indicating which time values are not
                        sampled by the cadence.
        """

        is_outside = Cadence.time_is_outside(self.long, time,
                                             inclusive=inclusive)
        if self.is_continuous:
            return is_outside

        tstep0 = self.long.tstep_at_time(time,
                                         remask=False,
                                         derivs=False,
                                         inclusive=inclusive)
        tstep0 = tstep0.int(top=self.shape[0])

        time0 = self.long.time_at_tstep(tstep0,
                                        remask=False,
                                        derivs=False)
        time1 = time - time0 + self._short_time0

        return is_outside.tvl_or(self.short.time_is_outside(time1,
                                                           inclusive=inclusive))

    #===========================================================================
    def time_shift(self, secs):
        """Construct a duplicate of this Cadence with all times shifted by given
        amount.

        Input:
            secs        the number of seconds to shift the time later.
        """

        return DualCadence(self.long.time_shift(secs), self.short)

    #===========================================================================
    def as_continuous(self):
        """Construct a shallow copy of this Cadence, forced to be continuous.

        For DualCadence, this is accomplished by forcing the stride of
        the short cadence to be continuous.
        """

        if self._short_duration >= self.long._max_tstride:
            # Note that self.long is already continuous
            return DualCadence(self.long, self.short.as_continuous())

        raise ValueError("short internal cadence cannot be extended to make " +
                         "this DualCadence continuous")

    #===========================================================================
    @staticmethod
    def for_array2d(samples, lines, tstart, texp, intersample_delay=0.,
                                                  interline_delay=None):
        """Alternative constructor for a DualCadence involving two Metronome
        classes, with streamlined input.

        Input:
            samples             number of samples (along fast axis).
            lines               number of lines (along slow axis).
            tstart              start time of observation in TDB seconds.
            texp                single-sample integration time in seconds.
            intersample_delay   deadtime in seconds between consecutive samples;
                                default 0.
            interline_delay     deadtime in seconds between consecutive lines,
                                i.e., the delay between the end of the last
                                sample integration on one line and the start of
                                the first sample integration on the next line.
                                If not specified, the interline_delay is assumed
                                to match the intersample_delay.
        """

        fast_cadence = Metronome(tstart, texp + intersample_delay, texp,
                                 samples)

        if interline_delay is None:
            interline_delay = intersample_delay

        long_texp = samples * texp + (samples-1) * intersample_delay
        long_stride = long_texp + interline_delay

        slow_cadence = Metronome(tstart, long_stride, long_texp, lines)

        return DualCadence(slow_cadence, fast_cadence)

################################################################################
# UNIT TESTS
################################################################################

import unittest
import numpy as np

class Test_DualCadence(unittest.TestCase):

    @staticmethod
    def meshgrid(*args):
        """A new Vector constructed by combining every possible set of
        components provided as a list of scalars. The returned Vector will
        have a shape defined by concatenating the shapes of all the arguments.

        This routine was stolen from the old array_ module and is not optimized
        for use with polymath.
        """

        scalars = []
        newshape = []
        dtype = "int"
        for arg in args:
            scalar = Scalar.as_scalar(arg)
            scalars.append(scalar)
            newshape += scalar.shape
            if scalar.vals.dtype.kind == "f":
                dtype = "float"

        buffer = np.empty(newshape + [len(args)], dtype=dtype)

        newaxes = []
        count = 0
        for scalar in scalars[::-1]:
            newaxes.append(count)
            count += len(scalar.shape)

        newaxes.reverse()

        for i in range(len(scalars)):
            scalars[i] = scalars[i].reshape(scalars[i].shape +
                                            newaxes[i] * (1,))

        reshaped = Qube.broadcast(*scalars)

        for i in range(len(reshaped)):
            buffer[...,i] = reshaped[i].vals

        return Vector(buffer)

    def runTest(self):

        import numpy.random as random

        np.random.seed(4305)

        # cad2d has shape (10,5)
        # cad1d has shape (50,)
        # We define them so that cad2d[i,j] = cad1d[5*i+j]

        # These should be equivalent except for 1-D vs. 2-D indexing

        # cad1d: 100-101, 102-103, 104-105, ... 198-199.
        cad1d = Metronome(100., 2., 1., 50)

        # 100-101, 110-111, 120-121, ... 190-191.
        # However, the continuous version is what's saved in the DualCadence
        # 100-110, 110-120, 120-130, ... 190-(200).
        long = Metronome(100., 10., 1., 10)

        # 0-1, 2-3, 4-5, 6-7, 8-9
        short = Metronome(0, 2., 1., 5)

        cad2d = DualCadence(long, short)

        self.assertEqual(cad1d.shape, (50,))
        self.assertEqual(cad2d.shape, (10,5))

        grid2d = Test_DualCadence.meshgrid(np.arange(10),np.arange(5))
        grid1d = 5. * grid2d.to_scalar(0) + grid2d.to_scalar(1)

        times1d = cad1d.time_at_tstep(grid1d, remask=False)
        times2d = cad2d.time_at_tstep(grid2d, remask=False)

        self.assertEqual(times1d, times2d)
        self.assertEqual(times1d.flatten(),
                         cad2d.time_at_tstep(grid2d.flatten(), remask=False))

        times1d = cad1d.time_at_tstep(grid1d, remask=True)
        times2d = cad2d.time_at_tstep(grid2d, remask=True)

        self.assertEqual(times1d, times2d)
        self.assertEqual(times1d.flatten(),
                         cad2d.time_at_tstep(grid2d.flatten()))

        range1d = cad1d.time_range_at_tstep(grid1d, remask=False)
        range2d = cad2d.time_range_at_tstep(grid2d, remask=False)

        self.assertEqual(range1d[0], range2d[0])
        self.assertEqual(range1d[1], range2d[1])

        range1d = cad1d.time_range_at_tstep(grid1d, remask=True)
        range2d = cad2d.time_range_at_tstep(grid2d, remask=True)

        self.assertEqual(range1d[0], range2d[0])
        self.assertEqual(range1d[1], range2d[1])

        test1d = cad1d.tstep_at_time(times1d, remask=False)
        test2d = cad2d.tstep_at_time(times2d, remask=False)

        self.assertEqual(test1d // 5, test2d.to_scalar(0))
        self.assertEqual(test1d %  5, test2d.to_scalar(1))

        test1d = cad1d.tstep_at_time(times1d, remask=True)
        test2d = cad2d.tstep_at_time(times2d, remask=True)

        self.assertEqual(test1d // 5, test2d.to_scalar(0))
        self.assertEqual(test1d %  5, test2d.to_scalar(1))

        time_seq = Scalar(np.arange(90,220) + 0.5)

        time_seq = Scalar(np.arange(90,220,10) + 0.5)
        test1d = cad1d.time_is_inside(time_seq)
        test2d = cad2d.time_is_inside(time_seq)
        self.assertTrue(test1d == test2d)

        # Test masked values
        tstep = Pair(((0,0),(1,1),(2,2)), [False,True,False])
        time = Scalar((100,110,120), [False,True,False])
        self.assertTrue(Boolean(cad2d.time_at_tstep(tstep).mask) ==
                        [False,True,False])
        self.assertTrue(Boolean(cad2d.tstep_at_time(time).to_scalar(0).mask) ==
                        [False,True,False])
        self.assertTrue(Boolean(cad2d.tstep_at_time(time).to_scalar(1).mask) ==
                        [False,True,False])
        self.assertTrue(Boolean(cad2d.time_is_inside(time).mask) == [False,True,False])
        self.assertTrue(Boolean(cad2d.time_range_at_tstep(tstep)[0].mask) ==
                        [False,True,False])
        self.assertTrue(Boolean(cad2d.time_range_at_tstep(tstep)[1].mask) ==
                        [False,True,False])

        # Random tsteps, using random floats
        values = np.random.rand(10,10,10,10,2)  # random values 0-1
        values[...,0] *= 12     # above 10 is out of range
        values[...,1] *= 7      # above 5 is out of range
        values -= 1             # shift so some values are negative
        # First index is now -1 to 11; second is -1 to 6.

        random2d = Pair(values)
        random1d = 5. * random2d.to_scalar(0).int() + random2d.to_scalar(1)

        times1d = cad1d.time_at_tstep(random1d, remask=False)
        times2d = cad2d.time_at_tstep(random2d, remask=False)
        self.assertTrue((abs(times1d - times2d) < 1.e-12).all())

        range1d = cad1d.time_range_at_tstep(random1d, remask=False)
        range2d = cad2d.time_range_at_tstep(random2d, remask=False)
        self.assertEqual(range1d[0], range2d[0])
        self.assertEqual(range1d[1], range2d[1])

        test1d = cad1d.tstep_at_time(times1d, remask=False)
        test2d = cad2d.tstep_at_time(times2d, remask=False)

        self.assertEqual(test1d // 5, test2d.to_scalar(0))
        self.assertTrue((abs(test1d % 5 - test2d.to_scalar(1)) < 1.e-12).all())

        # Make sure everything works with scalars
        for iter in range(1000):
            random1d = np.random.random()
            random2d = Vector((random1d//5, random1d%5))

            time1d = cad1d.time_at_tstep(random1d, remask=True)
            time2d = cad2d.time_at_tstep(random2d, remask=True)
            self.assertTrue(abs(time1d - time2d) < 1.e-12)

            range1d = cad1d.time_range_at_tstep(random1d, remask=True)
            range2d = cad2d.time_range_at_tstep(random2d, remask=True)
            self.assertEqual(range1d, range2d)

            test1d = cad1d.tstep_at_time(time1d, remask=True)
            test2d = cad2d.tstep_at_time(time2d, remask=True)

            self.assertEqual(test1d // 5, test2d.to_scalar(0))
            self.assertTrue(abs(test1d % 5 - test2d.to_scalar(1)) < 1.e-12)

        # time_shift()
        shifted = cad2d.time_shift(0.5)
        self.assertEqual(cad2d.time_at_tstep(grid2d, remask=True),
                         shifted.time_at_tstep(grid2d, remask=True) - 0.5)

        # tstride_at_tstep()
        self.assertEqual(cad2d.tstride_at_tstep(Pair((0,0))), Pair((10,2)))
        self.assertEqual(cad2d.tstride_at_tstep(Pair((5,3))), Pair((10,2)))

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
