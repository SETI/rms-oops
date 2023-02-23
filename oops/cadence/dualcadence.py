################################################################################
# oops/cadence/dualcadence.py: DualCadence subclass of class Cadence
################################################################################

from polymath               import Qube, Boolean, Scalar, Pair, Vector
from oops.cadence           import Cadence
from oops.cadence.metronome import Metronome

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

        self.long = long
        self.short = short.time_shift(-short.time[0])   # starts at time 0

        self.shape = self.long.shape + self.short.shape
        assert len(self.long.shape) == 1
        assert len(self.short.shape) == 1

        self.time = (self.long.time[0],
                     self.long.lasttime + self.short.time[1])
        self.midtime = (self.time[0] + self.time[1]) * 0.5
        self.lasttime = self.long.lasttime + self.short.lasttime

        self.is_continuous = (self.short.is_continuous and
                              self.time[0] >= self.long.max_tstride)

        self.is_unique = (self.short.is_unique and
                          self.short.time[0] <= self.long.min_tstride)

        self.min_tstride = self.short.min_tstride
        self.max_tstride = max(self.long.max_tstride - self.short.time[1],
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
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Scalar of times in seconds TDB.
        """

        tstep = Pair.as_pair(tstep, recursive=derivs)
        (long_tstep, short_tstep) = tstep.to_scalars()

        # Determine long start time
        long_time = self.long.time_range_at_tstep(long_tstep, remask=remask,
                                                  inclusive=inclusive)[0]

        # Determine short time
        short_time = self.short.time_at_tstep(short_tstep, remask=remask,
                                              derivs=derivs,
                                              inclusive=inclusive)

        return long_time + short_time

    #===========================================================================
    def time_range_at_tstep(self, tstep, remask=False, inclusive=True):
        """The range of times for the given time step.

        Input:
            tstep       a Pair of time step index values.
            remask      True to mask values outside the time limits.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        tstep = Pair.as_pair(tstep, recursive=False)
        (long_tstep, short_tstep) = tstep.to_scalars()

        # Determine long start time
        long_time0 = self.long.time_range_at_tstep(long_tstep, remask=remask,
                                                   inclusive=inclusive)[0]

        # Determine short time range
        short_times = self.short.time_range_at_tstep(short_tstep,
                                                     remask=remask,
                                                     inclusive=inclusive)

        return (long_time0 + short_times[0], long_time0 + short_times[1])

    #===========================================================================
    def tstep_at_time(self, time, remask=False, derivs=False, inclusive=True):
        """Time step for the given time.

        This method returns non-integer time steps.

        Input:
            time        a Scalar of times in seconds TDB.
            remask      True to mask time values not sampled within the cadence.
            derivs      True to include derivatives of tstep in the returned
                        time.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Pair of time step index values.
        """

        time = Scalar.as_scalar(time, recursive=derivs)

        # Determine long tstep
        # We need remask=False because the end time of each long cadence is
        # ignored; remask=True might mask some times incorrectly.
        tstep0 = self.long.tstep_range_at_time(time, remask=False,
                                               inclusive=inclusive)[0]

        # Determine short tstep
        time0 = self.long.time_at_tstep(tstep0, remask=remask,
                                        inclusive=inclusive)
        tstep1 = self.short.tstep_at_time(time - time0, remask=remask,
                                          derivs=derivs,
                                          inclusive=inclusive)

        # Revise long time step above the time limits
        if inclusive:
            tstep0[time.vals > self.time[1]] = self.shape[0]
        else:
            tstep0[time.vals >= self.time[1]] = self.shape[0]

        return Pair.from_scalars(tstep0, tstep1)

    #===========================================================================
    def tstep_range_at_time(self, time, remask=False, inclusive=True):
        """Integer range of time steps active at the given time.

        Input:
            time        a Scalar of times in seconds TDB.
            remask      True to mask time values not sampled within the cadence.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         (tstep_min, tstep_max)
            tstep_min   minimum Pair time step containing the given time.
            tstep_max   maximum Pair time step containing the given time
                        (inclusive).

        All returned indices will be in the allowed range for the cadence,
        inclusive, regardless of mask. If the time is not inside the cadence,
        tstep_max < tstep_min.
        """

        time = Scalar.as_scalar(time, recursive=False)

        # Find integer tsteps at or below the time values, unmasked.
        # Times before the start time map to tstep0_min = 0;
        # Times during or after the last time step map to shape[0]-1.
        tstep0_min = self.long.tstep_range_at_time(time, remask=False,
                                                   inclusive=False)[0]
        tstep0_max = tstep0_min + 1

        # Unique case is MUCH easier
        if self.is_unique:

            # Determine short tstep range
            time0 = self.long.time_at_tstep(tstep0_min, remask=remask,
                                            inclusive=inclusive)

            # Note: exclude the last moment of each short cadence
            # We address the last moment of the cadence overall below
            (tstep1_min,
             tstep1_max) = self.short.tstep_range_at_time(time - time0,
                                                          remask=remask,
                                                          inclusive=False)

            # Time step ranges outside time limits are already zero-length

            # Handle the last moment of the cadence
            if inclusive:
                mask = (time.vals == self.time[1]) & time.antimask
                tstep1_min[mask] = self.shape[1] - 1    # this also unmasks
                tstep1_max[mask] = self.shape[1]

        else:
            raise NotImplementedError('tstep_range_at_time is not implemented '+
                                      'for a non-unique DualCadence')

        # This step merges the tstep1 mask over the incomplete tstep0 masks
        tstep_min = Pair.from_scalars(tstep0_min, tstep1_min)
        tstep_max = Pair.from_scalars(tstep0_max, tstep1_max)
        return (tstep_min, tstep_max)

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

        time = Scalar.as_scalar(time, recursive=False)

        # Easier case
        if self.is_continuous:
            if inclusive:
                return (time < self.time[0]) | (time > self.time[1])
            else:
                return (time < self.time[0]) | (time >= self.time[1])

        # Determine long tstep
        tstep0 = self.long.tstep_range_at_time(time, inclusive=inclusive)[0]

        # Test for short tstep
        time0 = self.long.time_at_tstep(tstep0, inclusive=inclusive)
        return self.short.time_is_outside(time - time0, inclusive=inclusive)

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

        if self.time[0] >= self.long._max_tstride:
            return DualCadence(self.long, self.short.as_continuous())

        raise ValueError('short internal cadence cannot be extended to make ' +
                         'this DualCadence continuous')

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
        dtype = 'int'
        for arg in args:
            scalar = Scalar.as_scalar(arg)
            scalars.append(scalar)
            newshape += scalar.shape
            if scalar.vals.dtype.kind == 'f':
                dtype = 'float'

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

        np.random.seed(4305)

        # cad2d has shape (10,5)
        # cad1d has shape (50,)
        # We define them so that cad2d[i,j] = cad1d[5*i+j]

        # These should be equivalent except for 1-D vs. 2-D indexing

        # cad1d: 100-101, 102-103, 104-105, ... 198-199.
        cad1d = Metronome(100., 2., 1., 50)

        # 100-101, 110-111, 120-121, ... 190-191. (End time doesn't matter)
        long = Metronome(100., 10., 1., 10)

        # 0-1, 2-3, 4-5, 6-7, 8-9
        short = Metronome(0, 2., 1., 5)

        cad2d = DualCadence(long, short)
        case_dual_metronome(self, cad1d, cad2d)

def case_dual_metronome(self, cad1d, cad2d):

    self.assertEqual(cad1d.shape, (50,))
    self.assertEqual(cad2d.shape, (10,5))

    grid2d = Test_DualCadence.meshgrid(np.arange(10),np.arange(5))
    grid1d = 5. * grid2d.to_scalar(0) + grid2d.to_scalar(1)

    # time_at_tstep, grid
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

    # time_range_at_tstep
    range1d = cad1d.time_range_at_tstep(grid1d, remask=False)
    range2d = cad2d.time_range_at_tstep(grid2d, remask=False)
    self.assertEqual(range1d[0], range2d[0])
    self.assertEqual(range1d[1], range2d[1])

    range1d = cad1d.time_range_at_tstep(grid1d, remask=True)
    range2d = cad2d.time_range_at_tstep(grid2d, remask=True)
    self.assertEqual(range1d[0], range2d[0])
    self.assertEqual(range1d[1], range2d[1])

    # tstep_at_time
    test1d = cad1d.tstep_at_time(times1d, remask=False)
    test2d = cad2d.tstep_at_time(times2d, remask=False)
    self.assertEqual(test1d // 5, test2d.to_scalar(0))
    self.assertEqual(test1d %  5, test2d.to_scalar(1))

    test1d = cad1d.tstep_at_time(times1d, remask=True)
    test2d = cad2d.tstep_at_time(times2d, remask=True)
    self.assertEqual(test1d // 5, test2d.to_scalar(0))
    self.assertEqual(test1d %  5, test2d.to_scalar(1))

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

    # time_shift()
    shifted = cad2d.time_shift(0.5)
    self.assertEqual(cad2d.time_at_tstep(grid2d, remask=True),
                     shifted.time_at_tstep(grid2d, remask=True) - 0.5)

    # tstride_at_tstep()
    self.assertEqual(cad2d.tstride_at_tstep(Pair((0,0))), Pair((10,2)))
    self.assertEqual(cad2d.tstride_at_tstep(Pair((5,3))), Pair((10,2)))

    # Random tsteps, using random floats
    values = np.random.rand(10,10,10,10,2)  # random values 0-1
    values[...,0] *= 12     # above 10 is out of range
    values[...,1] *= 7      # above 5 is out of range
    values -= 1             # shift so some values are negative
    # First index is now -1 to 11; second is -1 to 6.

    random2d = Pair(values)
    random1d = 5. * random2d.to_scalar(0).as_int() + random2d.to_scalar(1)
    outside = ((values[...,0] < 0) | (values[...,0] >= 10) |
               (values[...,1] < 0) | (values[...,1] >= 5))

    times1d = cad1d.time_at_tstep(random1d, remask=False)
    times2d = cad2d.time_at_tstep(random2d, remask=False)
    self.assertTrue((abs(times1d - times2d)[~outside] < 1.e-13).all())

    range1d = cad1d.time_range_at_tstep(random1d, remask=False)
    range2d = cad2d.time_range_at_tstep(random2d, remask=False)
    self.assertEqual(range1d[0][~outside], range2d[0][~outside])
    self.assertEqual(range1d[1][~outside], range2d[1][~outside])

    test1d = cad1d.tstep_at_time(times1d, remask=False)
    test2d = cad2d.tstep_at_time(times2d, remask=False)
    self.assertEqual(test1d[~outside] // 5, test2d.to_scalar(0)[~outside])
    self.assertTrue((abs(test1d[~outside] % 5 - test2d[~outside].to_scalar(1)) < 1.e-13).all())

    times1d = cad1d.time_at_tstep(random1d, remask=True)
    times2d = cad2d.time_at_tstep(random2d, remask=True)
    self.assertTrue(np.all(outside == times2d.mask))
    self.assertTrue((abs(times1d - times2d)[~times2d.mask] < 1.e-13).all())

    range1d = cad1d.time_range_at_tstep(random1d, remask=False)
    range2d = cad2d.time_range_at_tstep(random2d, remask=False)
    self.assertEqual(range1d[0][~outside], range2d[0][~outside])
    self.assertEqual(range1d[1][~outside], range2d[1][~outside])

    test1d = cad1d.tstep_at_time(times1d, remask=False)
    test2d = cad2d.tstep_at_time(times2d, remask=False)
    self.assertEqual(test1d[~outside] // 5, test2d.to_scalar(0)[~outside])
    self.assertTrue((abs(test1d[~outside] % 5 - test2d[~outside].to_scalar(1)) < 1.e-13).all())

    # Make sure everything works with scalars
    for iter in range(100):
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

    # Random tsteps, using random floats, with derivs
    N = 200
    values = np.random.rand(N,2)  # random values 0-1
    values[...,0] *= 10
    values[...,1] *= 5

    mask = np.random.rand(N) < 0.2
    random2d = Pair(values, mask)

    array = np.random.randn(N,2)
    array[...,0] = 0.
    d_dt = Pair(array, mask)

    array = np.random.randn(N,2,2)
    array[...,0,:] = 0.
    d_dxy = Pair(array, mask, drank=1)
    random2d.insert_deriv('t', d_dt)
    random2d.insert_deriv('xy', d_dxy)

    random1d = 5. * random2d.to_scalar(0).as_int() + random2d.to_scalar(1)

    times1d = cad1d.time_at_tstep(random1d, derivs=True)
    times2d = cad2d.time_at_tstep(random2d, derivs=True)
    self.assertTrue((abs(times1d - times2d)[~mask] < 1.e-13).all())
    self.assertEqual(times2d.d_dt[~mask], d_dt.vals[...,1][~mask])
    self.assertEqual(times2d.d_dxy[~mask], d_dxy[~mask].vals[...,1,:])
    self.assertEqual(type(times2d.d_dt), Scalar)
    self.assertEqual(type(times2d.d_dxy), Scalar)
    self.assertEqual(times2d.d_dt.denom, ())
    self.assertEqual(times2d.d_dxy.denom, (2,))
    self.assertEqual(times2d.d_dt.shape, random2d.shape)
    self.assertEqual(times2d.d_dxy.shape, random2d.shape)

    test1d = cad1d.tstep_at_time(times1d, derivs=True)
    test2d = cad2d.tstep_at_time(times2d, derivs=True)
    self.assertEqual(test1d // 5, test2d.to_scalar(0))
    self.assertTrue((abs(test1d[~mask] % 5 - test2d[~mask].to_scalar(1)) < 1.e-13).all())
    self.assertEqual(test2d.d_dt[~mask], d_dt[~mask])
    self.assertEqual(test2d.d_dxy[~mask], d_dxy[~mask])
    self.assertEqual(type(test2d.d_dt), Pair)
    self.assertEqual(type(test2d.d_dxy), Pair)
    self.assertEqual(test2d.d_dt.denom, ())
    self.assertEqual(test2d.d_dxy.denom, (2,))
    self.assertEqual(test2d.d_dt.shape, random2d.shape)
    self.assertEqual(test2d.d_dxy.shape, random2d.shape)

    #### tstep_range_at_time, random

    time = 80 + 140 * np.random.rand(200)
    (tstep1a, tstep1b) = cad1d.tstep_range_at_time(time, remask=True)
    (tstep2a, tstep2b) = cad2d.tstep_range_at_time(time, remask=True)

    mask = tstep2a.mask
    self.assertTrue(np.all(tstep1a.vals[~mask]   == tstep1b.vals[~mask] - 1))
    self.assertTrue(np.all(tstep2a.vals[~mask,1] == tstep2b.vals[~mask,1] - 1))

    self.assertTrue(np.all(tstep1a.vals[tstep1a.mask] == tstep1b.vals[tstep1a.mask]))
    self.assertTrue(np.all(tstep2a.vals[tstep2a.mask,0] == tstep2b.vals[tstep2a.mask,0] - 1))
    self.assertTrue(np.all(tstep2a.vals[tstep2a.mask,1] == tstep2b.vals[tstep2a.mask,1]))

    test2a = 5.*tstep2a.vals[:,0] + tstep2a.vals[:,1]
    self.assertTrue(tstep1a[~mask] == test2a[~mask])

    test2b = 5.*(tstep2b.vals[:,0] - 1) + tstep2b.vals[:,1]
    self.assertTrue(tstep1b[~mask] == test2b[~mask])

    #### tstep_range_at_time, orderly, exclusive

    (tstep1a, tstep1b) = cad1d.tstep_range_at_time(time, remask=True, inclusive=False)
    (tstep2a, tstep2b) = cad2d.tstep_range_at_time(time, remask=True, inclusive=False)

    mask = tstep2a.mask
    self.assertTrue(np.all(tstep1a.vals[~mask] == tstep1b.vals[~mask] - 1))
    self.assertTrue(np.all(tstep2a.vals[~mask,1] == tstep2b.vals[~mask,1] - 1))

    self.assertTrue(np.all(tstep1a.vals[tstep1a.mask] == tstep1b.vals[tstep1a.mask]))
    self.assertTrue(np.all(tstep2a.vals[tstep2a.mask,1] == tstep2b.vals[tstep2a.mask,1]))

    test2a = 5.*tstep2a.vals[:,0] + tstep2a.vals[:,1]
    self.assertTrue(tstep1a[~mask] == test2a[~mask])

    test2b = 5.*(tstep2b.vals[:,0] - 1) + tstep2b.vals[:,1]
    self.assertTrue(tstep1b[~mask] == test2b[~mask])

    #### tstep_range_at_time, orderly, inclusive

    time = np.arange(80., 220., 0.125)
    (tstep1a, tstep1b) = cad1d.tstep_range_at_time(time, remask=True, inclusive=True)
    (tstep2a, tstep2b) = cad2d.tstep_range_at_time(time, remask=True, inclusive=True)

    mask = tstep2a.mask
    self.assertTrue(np.all(mask == tstep2b.mask))
    self.assertTrue(np.all(tstep1a.vals[~mask] == tstep1b.vals[~mask] - 1))
    self.assertTrue(np.all(tstep2a.vals[~mask,1] == tstep2b.vals[~mask,1] - 1))

    self.assertTrue(np.all(tstep1a.vals[tstep1a.mask] == tstep1b.vals[tstep1a.mask]))
    self.assertTrue(np.all(tstep2a.vals[tstep2a.mask,1] == tstep2b.vals[tstep2a.mask,1]))

    test2a = 5.*tstep2a.vals[:,0] + tstep2a.vals[:,1]
    self.assertTrue(tstep1a[~mask] == test2a[~mask])

    test2b = 5.*(tstep2b.vals[:,0] - 1) + tstep2b.vals[:,1]
    self.assertTrue(tstep1b[~mask] == test2b[~mask])

    #### out-of-range indices and times

    for i in range(-1,cad2d.shape[0]+2):
        self.assertEqual(cad2d.time_at_tstep((i,-0.5)),
                         cad2d.time_at_tstep((i, 0  )))
        self.assertEqual(cad2d.time_at_tstep((i, 5.5)),
                         cad2d.time_at_tstep((i, 5.0)))
        self.assertEqual(cad2d.time_range_at_tstep((i,-0.5)),
                         cad2d.time_range_at_tstep((i, 0  )))
        self.assertEqual(cad2d.time_range_at_tstep((i, 5.5)),
                         cad2d.time_range_at_tstep((i, 5.0)))

    for j in range(-1,cad2d.shape[1]+2):
        self.assertEqual(cad2d.time_at_tstep((-0.5, j)),
                         cad2d.time_at_tstep(( 0  , j)))
        self.assertEqual(cad2d.time_at_tstep((10.5, j)),
                         cad2d.time_at_tstep((10.0, j)))
        self.assertEqual(cad2d.time_range_at_tstep((-0.5, j)),
                         cad2d.time_range_at_tstep(( 0  , j)))
        self.assertEqual(cad2d.time_range_at_tstep((10.5, j)),
                         cad2d.time_range_at_tstep((10.0, j)))

    self.assertEqual(cad2d.tstep_at_time(99.), (0,0))
    self.assertEqual(cad2d.tstep_at_time(99., remask=True), (Scalar.MASKED, Scalar.MASKED))

    self.assertEqual(cad2d.tstep_at_time(190), (9,0))
    self.assertEqual(cad2d.tstep_at_time(190, inclusive=False), (9,0))
    self.assertEqual(cad2d.tstep_at_time(190, remask=True), (9,0))
    self.assertEqual(cad2d.tstep_at_time(190, inclusive=False,
                                              remask=True), (9,0))

    self.assertEqual(cad2d.tstep_at_time(198), (9,4))
    self.assertEqual(cad2d.tstep_at_time(198, inclusive=False), (9,4))
    self.assertEqual(cad2d.tstep_at_time(198, remask=True), (9,4))
    self.assertEqual(cad2d.tstep_at_time(198, inclusive=False,
                                              remask=True), (9,4))

    self.assertEqual(cad2d.tstep_at_time(199), (9,5))
    self.assertEqual(cad2d.tstep_at_time(199, inclusive=False), (10,5))
    self.assertEqual(cad2d.tstep_at_time(199, remask=True), (9,5))
    self.assertEqual(cad2d.tstep_at_time(199, inclusive=False,
                                              remask=True), (Scalar.MASKED, Scalar.MASKED))

    self.assertEqual(cad2d.tstep_at_time(200), (10,5))
    self.assertEqual(cad2d.tstep_at_time(200, remask=True), (Scalar.MASKED, Scalar.MASKED))

    self.assertEqual(cad2d.tstep_range_at_time(99.), ((0,0), (1,0)))
    self.assertEqual(cad2d.tstep_range_at_time(99., remask=True), (Pair.MASKED, Pair.MASKED))

    self.assertEqual(cad2d.tstep_range_at_time(190), ((9,0), (10,1)))
    self.assertEqual(cad2d.tstep_range_at_time(190, inclusive=False), ((9,0), (10,1)))
    self.assertEqual(cad2d.tstep_range_at_time(190, remask=True), ((9,0), (10,1)))
    self.assertEqual(cad2d.tstep_range_at_time(190, inclusive=False,
                                                    remask=True), ((9,0), (10,1)))

    self.assertEqual(cad2d.tstep_range_at_time(198), ((9,4), (10,5)))
    self.assertEqual(cad2d.tstep_range_at_time(198, inclusive=False), ((9,4), (10,5)))
    self.assertEqual(cad2d.tstep_range_at_time(198, remask=True), ((9,4), (10,5)))
    self.assertEqual(cad2d.tstep_range_at_time(198, inclusive=False,
                                                    remask=True), ((9,4), (10,5)))

    self.assertEqual(cad2d.tstep_range_at_time(199), ((9,4), (10,5)))
    self.assertEqual(cad2d.tstep_range_at_time(199, inclusive=False), ((9,4), (10,4)))
    self.assertEqual(cad2d.tstep_range_at_time(199, remask=True), ((9,4), (10,5)))
    self.assertEqual(cad2d.tstep_range_at_time(199, inclusive=False,
                                                    remask=True), (Pair.MASKED, Pair.MASKED))

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
