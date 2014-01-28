################################################################################
# oops_/cadence/dual.py: DualCadence subclass of class Cadence
#
# 7/2/12 MRS - created
################################################################################

import numpy as np
from oops_.array.all import *
from oops_.cadence.cadence_ import Cadence

class DualCadence(Cadence):
    """DualCadence is a Cadence subclass in which time steps are defined by a
    pair of cadences."""

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

        self.time = (self.long.time[0], self.long.lasttime + 
                                        self.short.time[1] - self.short.time[0])
        self.midtime = (self.time[0] + self.time[1]) * 0.5

        return

    def time_at_tstep(self, tstep, mask=False):
        """Returns the time associated with the given time step. This method
        supports non-integer step values.

        Input:
            tstep       a Scalar time step index or a Pair or Tuple of indices.
            mask        True to mask values outside the time limits.

        Return:         a Scalar of times in seconds TDB.
        """

        (long_step, short_step) = Tuple.as_tuple(tstep).as_scalars()

        return (self.long.time_at_tstep(long_step.int(), mask) +
                self.short.time_at_tstep(short_step, mask) - self.short.time[0])

    def time_range_at_tstep(self, tstep, mask=False):
        """Returns the range of time associated with the given integer time
        step index.

        Input:
            indices     a Scalar time step index or a Pair or Tuple of indices.
            mask        True to mask values outside the time limits.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        (long_step, short_step) = Tuple.as_tuple(tstep).as_scalars()

        long_ref = self.long.time_at_tstep(long_step.int(),
                                           mask) - self.short.time[0]
        (short0, short1) = self.short.time_range_at_tstep(short_step, mask)

        return (long_ref + short0, long_ref + short1)

    def tstep_at_time(self, time, mask=False):
        """Returns a the Scalar time step index or a Pair or Tuple of indices
        associated with a time in seconds TDB.

        Input:
            time        a Scalar of times in seconds TDB.
            mask        True to mask time values not sampled within the cadence.

        Return:         a Scalar, Pair or Tuple of time step indices.
        """

        tstep0 = self.long.tstep_at_time(time).int()

        time1 = (self.short.time[0] +
                 time - self.long.time_range_at_tstep(tstep0)[0])
        tstep1 = self.short.tstep_at_time(time1, mask=mask)

        return Pair.from_scalars(tstep0,tstep1)

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

        tstep0 = self.long.tstep_at_time(time)

        time1 = (self.short.time[0] +
                 time - self.long.time_range_at_tstep(tstep0)[0])
        return self.short.time_is_inside(time1, inclusive=inclusive)

    def time_shift(self, secs):
        """Returns a duplicate of the given cadence, with all times shifted by
        a specified number of seconds."

        Input:
            secs        the number of seconds to shift the time later.
        """

        return DualCadence(self.long.time_shift(secs), self.short)

    def as_continuous(self):
        """Returns a shallow copy of the given cadence, with equivalent strides
        but with the property that the cadence is continuous.
        """

        return DualCadence(self.long, self.short.as_continuous())

################################################################################
# UNIT TESTS
################################################################################

import unittest
import numpy.random as random
from oops_.cadence.metronome import Metronome

class Test_DualCadence(unittest.TestCase):

    def runTest(self):

        # These should be equivalent except for 1-D vs. 2-D indexing
        cad1d = Metronome(100., 2., 1., 50)

        long = Metronome(100., 10., 1., 10)
        short = Metronome(0, 2., 1., 5)
        cad2d = DualCadence(long,short)

        self.assertEqual(cad1d.shape, (50,))
        self.assertEqual(cad2d.shape, (10,5))

        grid2d = Tuple.meshgrid(np.arange(10),np.arange(5))
        grid1d = 5. * grid2d.as_scalar(0) + grid2d.as_scalar(1)

        times1d = cad1d.time_at_tstep(grid1d, mask=True)
        times2d = cad2d.time_at_tstep(grid2d, mask=True)

        self.assertEqual(times1d, times2d)
        self.assertEqual(times1d.flatten(),
                         cad2d.time_at_tstep(grid2d.flatten()))

        range1d = cad1d.time_range_at_tstep(grid1d)
        range2d = cad2d.time_range_at_tstep(grid2d)

        self.assertEqual(range1d, range2d)

        test1d = cad1d.tstep_at_time(times1d)
        test2d = cad2d.tstep_at_time(times2d)

        self.assertEqual(test1d // 5, test2d.as_scalar(0))
        self.assertEqual(test1d %  5, test2d.as_scalar(1))

        # Random tsteps
        values = np.random.rand(10,10,10,10,2)
        values[...,0] *= 12
        values[...,0] *= 7
        values -= 1
        random2d = Tuple(values)
        random1d = 5. * random2d.as_scalar(0).int() + random2d.as_scalar(1)

        times1d = cad1d.time_at_tstep(random1d)
        times2d = cad2d.time_at_tstep(random2d)
        self.assertTrue(abs(times1d - times2d) < 1.e-12)

        range1d = cad1d.time_range_at_tstep(random1d)
        range2d = cad2d.time_range_at_tstep(random2d)
        self.assertEqual(range1d, range2d)

        test1d = cad1d.tstep_at_time(times1d)
        test2d = cad2d.tstep_at_time(times2d)

        self.assertEqual(test1d // 5, test2d.as_scalar(0))
        self.assertTrue(abs(test1d % 5 - test2d.as_scalar(1)) < 1.e-12)

        # Make sure everything works with scalars
        for iter in range(100):
            random1d = np.random.random()
            random2d = Tuple((random1d//5, random1d%5))

            time1d = cad1d.time_at_tstep(random1d)
            time2d = cad2d.time_at_tstep(random2d)
            self.assertTrue(abs(time1d - time2d) < 1.e-12)

            range1d = cad1d.time_range_at_tstep(random1d)
            range2d = cad2d.time_range_at_tstep(random2d)
            self.assertEqual(range1d, range2d)

            test1d = cad1d.tstep_at_time(time1d)
            test2d = cad2d.tstep_at_time(time2d)

            self.assertEqual(test1d // 5, test2d.as_scalar(0))
            self.assertTrue(abs(test1d % 5 - test2d.as_scalar(1)) < 1.e-12)

        # time_shift()
        shifted = cad2d.time_shift(0.5)
        self.assertEqual(cad2d.time_at_tstep(grid2d),
                         shifted.time_at_tstep(grid2d) - 0.5)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

