################################################################################
# oops/cadence_/dual.py: DualCadence subclass of class Cadence
################################################################################

from polymath import *
from oops.cadence_.cadence import Cadence

class DualCadence(Cadence):
    """DualCadence is a Cadence subclass in which time steps are defined by a pair of cadences."""

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
                                        self.short.time[1] -
                                        self.short.time[0])
        self.midtime = (self.time[0] + self.time[1]) * 0.5
        
        # These are arbitrary definitions and may need to be changed in the future
        self.is_continuous = self.short.is_continuous
        self.lasttime = self.long.lasttime
        
        return

    def time_at_tstep(self, tstep, mask=True):
        """Return the time(s) associated with the given time step(s).
        
        This method supports non-integer step values.

        Input:
            tstep       a Scalar time step index or a Pair of indices.
            mask        True to mask values outside the time limits.

        Return:         a Scalar of times in seconds TDB.
        """

        (long_step, short_step) = Vector.as_vector(tstep).to_scalars()

        return (self.long.time_at_tstep(long_step.int(), mask) +
                self.short.time_at_tstep(short_step, mask) -
                self.short.time[0])

    def time_range_at_tstep(self, tstep, mask=True):
        """Return the range of time(s) for the given integer time step(s).

        Input:
            indices     a Scalar time step index or a Pair of indices.
            mask        True to mask values outside the time limits.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        (long_step, short_step) = Vector.as_vector(tstep).to_scalars()

        long_ref = self.long.time_at_tstep(long_step.int(),
                                           mask) - self.short.time[0]
        (short0, short1) = self.short.time_range_at_tstep(short_step, mask)

        return (long_ref + short0, long_ref + short1)

    def tstep_at_time(self, time, mask=True):
        """Return the time step(s) for given time(s).

        This method supports non-integer time values.

        Input:
            time        a Scalar of times in seconds TDB.
            mask        True to mask time values not sampled within the cadence.

        Return:         a Scalar or Pair of time step indices.
        """

        tstep0 = self.long.tstep_at_time(time, mask=mask).int()

        time1 = (self.short.time[0] +
                 time - self.long.time_range_at_tstep(tstep0, mask=mask)[0])
        tstep1 = self.short.tstep_at_time(time1, mask=mask)

        return Pair.from_scalars(tstep0,tstep1)

    def time_is_inside(self, time, inclusive=True):
        """Return which time(s) fall inside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to include the end moment of a time interval;
                        False to exclude.

        Return:         a Boolean array indicating which time values are
                        sampled by the cadence. A masked time results in a
                        value of False, not a masked Boolean.
        """

        tstep0 = self.long.tstep_at_time(time)

        time1 = (self.short.time[0] +
                 time - self.long.time_range_at_tstep(tstep0)[0])
        return (self.long.time_is_inside(time, inclusive=inclusive) &
                self.short.time_is_inside(time1, inclusive=inclusive))

    def time_shift(self, secs):
        """Return a duplicate with all times shifted by given amount."

        Input:
            secs        the number of seconds to shift the time later.
        """

        return DualCadence(self.long.time_shift(secs), self.short)

    def as_continuous(self):
        """Return a shallow copy forced to be continuous.
        
        For DualCadence, this is accomplished by forcing the stride of
        the short cadence to be continuous.
        """

        return DualCadence(self.long, self.short.as_continuous())

################################################################################
# UNIT TESTS
################################################################################

import unittest
import numpy as np

class Test_DualCadence(unittest.TestCase):

    @staticmethod
    def meshgrid(*args):
        """Returns a new Vector constructed by combining every possible set of
        components provided as a list of scalars. The returned Vector will have
        a shape defined by concatenating the shapes of all the arguments.
        
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
            if scalar.vals.dtype.kind == "f": dtype = "float"

        buffer = np.empty(newshape + [len(args)], dtype=dtype)

        newaxes = []
        count = 0
        for scalar in scalars[::-1]:
            newaxes.append(count)
            count += len(scalar.shape)

        newaxes.reverse()

        for i in range(len(scalars)):
            scalars[i] = scalars[i].reshape(list(scalars[i].shape) +
                                            newaxes[i] * [1])

        reshaped = Qube.broadcast(*scalars)

        for i in range(len(reshaped)):
            buffer[...,i] = reshaped[i].vals

        return Vector(buffer)

    def runTest(self):

        import numpy.random as random
        from oops.cadence_.metronome import Metronome

        # These should be equivalent except for 1-D vs. 2-D indexing
        # Note that we don't test the invalid data cases because nothing
        # in the implementation depends on them
        cad1d = Metronome(100., 2., 1., 50)

        long = Metronome(100., 10., 1., 10)
        short = Metronome(0, 2., 1., 5)
        cad2d = DualCadence(long,short)

        self.assertEqual(cad1d.shape, (50,))
        self.assertEqual(cad2d.shape, (10,5))

        grid2d = Test_DualCadence.meshgrid(np.arange(10),np.arange(5))
        grid1d = 5. * grid2d.to_scalar(0) + grid2d.to_scalar(1)

        times1d = cad1d.time_at_tstep(grid1d, mask=False)
        times2d = cad2d.time_at_tstep(grid2d, mask=False)

        self.assertEqual(times1d, times2d)
        self.assertEqual(times1d.flatten(),
                         cad2d.time_at_tstep(grid2d.flatten(), mask=False))

        times1d = cad1d.time_at_tstep(grid1d)
        times2d = cad2d.time_at_tstep(grid2d)

        self.assertEqual(times1d, times2d)
        self.assertEqual(times1d.flatten(),
                         cad2d.time_at_tstep(grid2d.flatten()))

        range1d = cad1d.time_range_at_tstep(grid1d, mask=False)
        range2d = cad2d.time_range_at_tstep(grid2d, mask=False)

        self.assertEqual(range1d, range2d)

        range1d = cad1d.time_range_at_tstep(grid1d)
        range2d = cad2d.time_range_at_tstep(grid2d)

        self.assertEqual(range1d, range2d)

        test1d = cad1d.tstep_at_time(times1d, mask=False)
        test2d = cad2d.tstep_at_time(times2d, mask=False)

        self.assertEqual(test1d // 5, test2d.to_scalar(0))
        self.assertEqual(test1d %  5, test2d.to_scalar(1))

        test1d = cad1d.tstep_at_time(times1d)
        test2d = cad2d.tstep_at_time(times2d)

        self.assertEqual(test1d // 5, test2d.to_scalar(0))
        self.assertEqual(test1d %  5, test2d.to_scalar(1))

        # time_is_inside differs in the two cases at exactly time=200
        # so we carefully skip over that case
        time_seq = Scalar(np.arange(90,220)+0.5)
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
        self.assertTrue(cad2d.time_is_inside(time) == [True,False,True])
        self.assertTrue(Boolean(cad2d.time_range_at_tstep(tstep)[0].mask) ==
                        [False,True,False])
        self.assertTrue(Boolean(cad2d.time_range_at_tstep(tstep)[1].mask) ==
                        [False,True,False])

        # Random tsteps
        values = np.random.rand(10,10,10,10,2)
        values[...,0] *= 12
        values[...,1] *= 7
        values -= 1
        random2d = Vector(values)
        random1d = 5. * random2d.to_scalar(0).int() + random2d.to_scalar(1)

        times1d = cad1d.time_at_tstep(random1d, mask=False)
        times2d = cad2d.time_at_tstep(random2d, mask=False)
        self.assertTrue((abs(times1d - times2d) < 1.e-12).all())

        range1d = cad1d.time_range_at_tstep(random1d, mask=False)
        range2d = cad2d.time_range_at_tstep(random2d, mask=False)
        self.assertEqual(range1d, range2d)

        test1d = cad1d.tstep_at_time(times1d, mask=False)
        test2d = cad2d.tstep_at_time(times2d, mask=False)
        
        self.assertEqual(test1d // 5, test2d.to_scalar(0))
        self.assertTrue((abs(test1d % 5 - test2d.to_scalar(1)) < 1.e-12).all())

        # Make sure everything works with scalars
        for iter in range(1000):
            random1d = np.random.random()
            random2d = Vector((random1d//5, random1d%5))

            time1d = cad1d.time_at_tstep(random1d)
            time2d = cad2d.time_at_tstep(random2d)
            self.assertTrue(abs(time1d - time2d) < 1.e-12)

            range1d = cad1d.time_range_at_tstep(random1d)
            range2d = cad2d.time_range_at_tstep(random2d)
            self.assertEqual(range1d, range2d)

            test1d = cad1d.tstep_at_time(time1d)
            test2d = cad2d.tstep_at_time(time2d)

            self.assertEqual(test1d // 5, test2d.to_scalar(0))
            self.assertTrue(abs(test1d % 5 - test2d.to_scalar(1)) < 1.e-12)

        # time_shift()
        shifted = cad2d.time_shift(0.5)
        self.assertEqual(cad2d.time_at_tstep(grid2d),
                         shifted.time_at_tstep(grid2d) - 0.5)

        # tstride_at_tstep()
        self.assertEqual(cad2d.tstride_at_tstep(Pair((0,0))), Pair((10,2)))
        self.assertEqual(cad2d.tstride_at_tstep(Pair((5,3))), Pair((10,2)))
        
########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

