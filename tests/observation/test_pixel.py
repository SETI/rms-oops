################################################################################
# tests/observation/test_pixel.py
################################################################################

import numpy as np
import unittest

from polymath import Scalar, Pair

from oops.cadence     import Metronome
from oops.fov         import FlatFOV
from oops.observation import Pixel


class Test_Pixel(unittest.TestCase):

    def runTest(self):

        fov = FlatFOV((0.001,0.001), (1,1))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        obs = Pixel(axes=('t'),
                    cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Scalar([(0,),(1,),(20,),(21,)])
        indices_ = indices.copy()
        indices_.vals[indices_.vals == 20] -= 1         # clip the top

        # uvt() with remask == False
        (uv, time) = obs.uvt(indices)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, cadence.time_at_tstep(indices))
        self.assertEqual(uv, (0.5,0.5))

        # uvt() with remask == True
        (uv, time) = obs.uvt(indices, remask=True)

        self.assertTrue(np.all(uv.mask == np.array([3*[[False]] + [[True]]])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:3], cadence.tstride * indices[:3])
        self.assertEqual(uv[:3], (0.5,0.5))

        # uvt_range() with remask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min, (0,0))
        self.assertEqual(uv_max, (1,1))

        self.assertEqual(time_min, cadence.time_range_at_tstep(indices_)[0])
        self.assertEqual(time_max, time_min + cadence.texp)

        # uvt_range() with remask == False, new indices
        non_ints = indices + 0.2
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min, (0,0))
        self.assertEqual(uv_max, (1,1))

        self.assertEqual(time_min, cadence.time_range_at_tstep(non_ints)[0])
        self.assertEqual(time_max, time_min + cadence.texp)

        # uvt_range() with remask == True, new indices
        non_ints = indices + 0.2
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints, remask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(2*[[False]] + 2*[[True]])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min[:2], (0,0))
        self.assertEqual(uv_max[:2], (1,1))
        self.assertEqual(time_min[:2], indices[:2] * cadence.tstride)
        self.assertEqual(time_max[:2], time_min[:2] + cadence.texp)

        # time_range_at_uv() with remask == False
        uv = Pair([(0,0),(0,1),(1,0),(1,1),(1,2)])

        (time0, time1) = obs.time_range_at_uv(uv)

        self.assertEqual(time0, obs.time[0])
        self.assertEqual(time1, obs.time[1])

        # time_range_at_uv() with remask == True
        (time0, time1) = obs.time_range_at_uv(uv, remask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == time0.mask))
        self.assertEqual(time0[:4], obs.time[0])
        self.assertEqual(time1[:4], obs.time[1])

        ####################################

        # Alternative axis order ('a','t')

        fov = FlatFOV((0.001,0.001), (1,1))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        obs = Pixel(axes=('a','t'),
                    cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Pair([(0,0),(1,1),(0,20,),(1,21)])
        indices_ = indices.copy()
        indices_.vals[indices_.vals == 20] -= 1         # clip the top

        # uvt() with remask == False
        (uv,time) = obs.uvt(indices)

        self.assertFalse(uv.mask)
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time.without_mask(),
                         cadence.time_at_tstep(indices.to_scalar(1)))
        self.assertEqual(uv, (0.5,0.5))

        # uvt() with remask == True
        (uv,time) = obs.uvt(indices, remask=True)

        self.assertTrue(np.all(uv.mask == np.array(3*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:3], cadence.tstride * indices[:3].to_scalar(1))
        self.assertEqual(uv[:3], (0.5,0.5))

        # uvt_range() with remask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, (0,0))
        self.assertEqual(uv_max, (1,1))

        self.assertEqual(time_min, cadence.time_range_at_tstep(indices.to_scalar(1))[0])
        self.assertEqual(time_max, time_min + cadence.texp)

        # uvt_range() with remask == False, new indices
        non_ints = indices + (0.2,0.9)
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints)

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, (0,0))
        self.assertEqual(uv_max, (1,1))

        self.assertEqual(time_min, cadence.time_range_at_tstep(indices.to_scalar(1))[0])
        self.assertEqual(time_max, time_min + cadence.texp)

        # uvt_range() with remask == True, new indices
        non_ints = indices + (0.2,0.2)
        (uv_min, uv_max, time_min,
                         time_max) = obs.uvt_range(non_ints, remask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(2*[False] + 2*[True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min[:2], (0,0))
        self.assertEqual(uv_max[:2], (1,1))
        self.assertEqual(uv_min[2:], Pair.MASKED)
        self.assertEqual(uv_max[2:], Pair.MASKED)

        self.assertEqual(time_min[:2], indices.to_scalar(1)[:2] * cadence.tstride)
        self.assertEqual(time_max[:2], time_min[:2] + cadence.texp)

        # time_range_at_uv() with remask == False
        uv = Pair([(0,0),(0,1),(1,0),(1,1),(1,2)])

        (time0, time1) = obs.time_range_at_uv(uv)
        self.assertEqual(time0, obs.time[0])
        self.assertEqual(time1, obs.time[1])

        # time_range_at_uv() with remask == True
        (time0, time1) = obs.time_range_at_uv(uv, remask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == time0.mask))
        self.assertEqual(time0[:4], obs.time[0])
        self.assertEqual(time1[:4], obs.time[1])

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
