################################################################################
# test/cadence/test_dualcadence.py
################################################################################

import numpy as np
import unittest

from polymath import Qube, Boolean, Scalar, Pair, Vector
import oops

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
        cad1d = oops.cadence.Metronome(100., 2., 1., 50)

        # 100-101, 110-111, 120-121, ... 190-191. (End time doesn't matter)
        long = oops.cadence.Metronome(100., 10., 1., 10)

        # 0-1, 2-3, 4-5, 6-7, 8-9
        short = oops.cadence.Metronome(0, 2., 1., 5)

        cad2d = oops.cadence.DualCadence(long, short)
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
    for count in range(100):
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
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
