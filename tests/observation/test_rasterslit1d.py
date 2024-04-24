################################################################################
# tests/observation/test_rasterslit1d.py
################################################################################

import numpy as np
import unittest

from polymath         import Scalar, Pair
from oops.cadence     import Metronome
from oops.fov         import FlatFOV
from oops.observation import RasterSlit1D


class Test_RasterSlit1D(unittest.TestCase):

    def runTest(self):

        ############################################
        # Continuous 2-D observation
        # First axis = U and T with length 10
        # Second axis ignored
        ############################################

        fov = FlatFOV((0.001,0.001), (10,1))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
        obs = RasterSlit1D(axes=('ut','a'), cadence=cadence,
                           fov=fov, path='SSB', frame='J2000')

        indices = Pair([(0,0),(10,0),(11,0)])
        indices_ = indices.copy()   # clipped at top
        indices_.vals[:,0][indices_.vals[:,0] == 10] -= 1

        # uvt() with remask == False
        (uv, time) = obs.uvt(indices)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, cadence.time_at_tstep(indices.to_scalar(0)))
        self.assertEqual(uv.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv.to_scalar(1), 0.5)

        # uvt() with remask == True
        (uv, time) = obs.uvt(indices, remask=True)

        self.assertTrue(np.all(uv.mask == np.array(2*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:2], cadence.tstride * indices.to_scalar(0)[:2])
        self.assertEqual(uv[:2].to_scalar(0), indices[:2].to_scalar(0))
        self.assertEqual(uv[:2].to_scalar(1), 0.5)

        # uvt() with remask == True, new indices
        non_ints = indices + (0.2,0.9)
        (uv, time) = obs.uvt(non_ints, remask=True)

        self.assertTrue(np.all(uv.mask == np.array([False] + 2*[True])))
        self.assertTrue(np.all(time.mask == uv.mask))

        # uvt_range() with remask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv_min.to_scalar(1), 0)
        self.assertEqual(uv_max.to_scalar(0), indices.to_scalar(0) + 1)
        self.assertEqual(uv_max.to_scalar(1), 1)
        self.assertEqual(time_min, cadence.time_range_at_tstep(indices.to_scalar(0))[0])
        self.assertEqual(time_max, time_min + 10.)

        # uvt_range() with remask == True
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices,
                                                             remask=True)

        self.assertTrue(np.all(uv_min.mask == np.array([False, False, True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min.to_scalar(0)[:2], indices_.to_scalar(0)[:2])
        self.assertEqual(uv_min.to_scalar(1)[:2], 0)
        self.assertEqual(uv_max.to_scalar(0)[:2], indices_.to_scalar(0)[:2] + 1)
        self.assertEqual(uv_max.to_scalar(1)[:2], 1)
        self.assertEqual(time_min[:2], cadence.tstride*indices_.to_scalar(0)[:2])
        self.assertEqual(time_max[:2], time_min[:2] + cadence.texp)

        self.assertEqual(uv_min[2], Pair.MASKED)
        self.assertEqual(time_min[2], Scalar.MASKED)
        self.assertEqual(time_min[2], Scalar.MASKED)

        # time_range_at_uv() with remask == False
        uv = Pair([(0,0),(0,20),(10,0),(10,20),(11,21)])
        uv_ = uv.copy()
        uv_.vals[:,0][uv_.vals[:,0] == 10] -= 1

        (time0, time1) = obs.time_range_at_uv(uv)

        self.assertEqual(time0, cadence.time_range_at_tstep(uv_.to_scalar(0))[0])
        self.assertEqual(time1, time0 + cadence.texp)

        # time_range_at_uv() with remask == True
        (time0, time1) = obs.time_range_at_uv(uv, remask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == 4*[False] + [True]))
        self.assertEqual(time0[:4], cadence.tstride * uv_.to_scalar(0)[:4])
        self.assertEqual(time1[:4], time0[:4] + cadence.texp)

        ############################################################
        # Alternative axis order ('a', 'vt')
        # Second axis = V and T with length 10
        # Discontinuous time sampling [0-8], [10-18], ..., [90-98]
        # First axis ignored
        ############################################################

        fov = FlatFOV((0.001,0.001), (1,10))
        cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
        obs = RasterSlit1D(axes=('a','vt'), cadence=cadence,
                           fov=fov, path='SSB', frame='J2000')

        indices = Pair([(0,0),(0,9),(0,10),(0,11)])
        indices_ = indices.copy()   # clipped at top
        indices_.vals[:,1][indices_.vals[:,1] == 10] -= 1

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv.to_scalar(0), 0.5)
        self.assertEqual(uv.to_scalar(1), indices.to_scalar(1))
        self.assertEqual(time, [0,90,98,98])

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min.to_scalar(0), 0)
        self.assertEqual(uv_min.to_scalar(1), indices_.to_scalar(1))
        self.assertEqual(uv_max.to_scalar(0), 1)
        self.assertEqual(uv_max.to_scalar(1), indices_.to_scalar(1) + 1)
        self.assertEqual(time_min, cadence.time_range_at_tstep(indices_.to_scalar(1))[0])
        self.assertEqual(time_max, time_min + cadence.texp)

        uv = Pair([(11,0),(11,9),(11,10),(11,11)])
        uv_ = uv.copy()
        uv_.vals[:,1][uv_.vals[:,1] == 10] -= 1

        (time0, time1) = obs.time_range_at_uv(uv)

        self.assertEqual(time0, cadence.time_range_at_tstep(uv_.to_scalar(1))[0])
        self.assertEqual(time1, time0 + cadence.texp)

        ############################################################
        # Similar to above but 1-D observation
        # First axis = V and T with length 10
        # Discontinuous time sampling [0-8], [10-18], ..., [90-98]
        ############################################################

        fov = FlatFOV((0.001,0.001), (1,10))
        cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
        obs = RasterSlit1D(axes=('vt',), cadence=cadence,
                           fov=fov, path='SSB', frame='J2000')

        indices = Scalar([0,9,10,11])
        indices_ = indices.copy()   # clipped at top
        indices_.vals[indices_.vals == 10] -= 1

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv.to_scalar(0), 0.5)
        self.assertEqual(uv.to_scalar(1), indices)
        self.assertEqual(time, [0,90,98,98])

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min.to_scalar(0), 0)
        self.assertEqual(uv_min.to_scalar(1), indices_)
        self.assertEqual(uv_max.to_scalar(0), 1)
        self.assertEqual(uv_max.to_scalar(1), indices_ + 1)
        self.assertEqual(time_min, cadence.time_range_at_tstep(indices_)[0])
        self.assertEqual(time_max, time_min + cadence.texp)

        uv = Pair([(11,0),(11,9),(11,10),(11,11)])
        uv_ = uv.copy()
        uv_.vals[:,1][uv_.vals[:,1] == 10] -= 1

        (time0, time1) = obs.time_range_at_uv(uv)

        self.assertEqual(time0, cadence.time_range_at_tstep(uv_.to_scalar(1))[0])
        self.assertEqual(time1, time0 + cadence.texp)

        ############################################################
        # Alternative axis order ('ut',), 1-D
        # First axis = U and T with length 10
        # Discontinuous time sampling [0-8], [10-18], ..., [90-98]
        ############################################################

        fov = FlatFOV((0.001,0.001), (10,1))
        cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
        obs = RasterSlit1D(axes=('ut',), cadence=cadence,
                           fov=fov, path='SSB', frame='J2000')

        self.assertEqual(obs.time[1], 98.)

        self.assertEqual(obs.uvt(0,True)[1],    0.)
        self.assertEqual(obs.uvt(5,True)[1],   50.)
        self.assertEqual(obs.uvt(5.5,True)[1], 54.)
        self.assertEqual(obs.uvt(9.5,True)[1], 94.)
        self.assertEqual(obs.uvt(10.,True)[1], 98.)
        self.assertTrue(obs.uvt(10.001,True)[1].mask)

        eps = 1.e-14
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((6.     ),True)[0] - (6.0,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.2    ),True)[0] - (6.2,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.4    ),True)[0] - (6.4,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.6    ),True)[0] - (6.6,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.8    ),True)[0] - (6.8,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((7.     ),True)[0] - (7.0,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((10     ),True)[0] - (10.,0.5)) < delta)
        self.assertTrue(obs.uvt(10.+eps,True)[0].mask)

        indices = Scalar([10-eps, 10, 10+eps])

        (uv,t) = obs.uvt(indices, remask=True)
        self.assertTrue(np.all(t.mask == np.array(2*[False] + [True])))

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
