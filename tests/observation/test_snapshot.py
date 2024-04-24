################################################################################
# tests/observation/test_snapshot.py
################################################################################

import numpy as np
import unittest

from polymath         import Pair, Vector
from oops.fov         import FlatFOV
from oops.observation import Snapshot


class Test_Snapshot(unittest.TestCase):

    def runTest(self):


        fov = FlatFOV((0.001,0.001), (10,20))
        obs = Snapshot(('u','v'), tstart=98., texp=2.,
                       fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0.,0.),(0.,20.),(10.,0.),(10.,20.),(10.,21.)])
        indices_ = indices.copy()
        indices_.vals[:,0][indices.vals[:,0] == 10] -= 1
        indices_.vals[:,1][indices.vals[:,1] == 20] -= 1

        # uvt() with remask == False
        (uv,time) = obs.uvt(indices)

        self.assertFalse(uv.mask)
        self.assertFalse(time.mask)
        self.assertEqual(time, 99.)
        self.assertEqual(uv, Pair.as_pair(indices))

        # uvt() with remask == True
        (uv,time) = obs.uvt(indices, remask=True)

        self.assertTrue(np.all(uv.mask == np.array(4*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:4], 99.)
        self.assertEqual(uv[:4], Pair.as_pair(indices)[:4])

        # uvt_range() with remask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, Pair.as_pair(indices_))
        self.assertEqual(uv_max, Pair.as_pair(indices_) + (1,1))
        self.assertEqual(time_min,  98.)
        self.assertEqual(time_max, 100.)

        # uvt_range() with remask == False, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9))

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, Pair.as_pair(indices))
        self.assertEqual(uv_max, Pair.as_pair(indices) + (1,1))
        self.assertEqual(time_min,  98.)
        self.assertEqual(time_max, 100.)

        # uvt_range() with remask == True, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9),
                                                             remask=True)
        self.assertTrue(np.all(uv_min.mask == [False] + 4*[True]))
        self.assertTrue(np.all(uv_min.mask == uv_max.mask))
        self.assertTrue(np.all(uv_min.mask == time_min.mask))
        self.assertTrue(np.all(uv_min.mask == time_max.mask))

        self.assertEqual(uv_min[0], Pair.as_pair(indices)[0])
        self.assertEqual(uv_max[0], (Pair.as_pair(indices) + (1,1))[0])
        self.assertEqual(time_min[0],  98.)
        self.assertEqual(time_max[0], 100.)

        # time_range_at_uv() with remask == False
        uv_pair = Pair([(0.,0.),(0.,20.),(10.,0.),(10.,20.),(10.,21.)])

        (time0, time1) = obs.time_range_at_uv(uv_pair)

        self.assertEqual(time0,  98.)
        self.assertEqual(time1, 100.)

        # time_range_at_uv() with remask == True
        (time0, time1) = obs.time_range_at_uv(uv_pair, remask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == 4*[False] + [True]))
        self.assertEqual(time0[:4],  98.)
        self.assertEqual(time1[:4], 100.)

        # Alternative axis order ('v','u')
        obs = Snapshot(('v','u'), tstart=98., texp=2.,
                       fov=fov, path='SSB', frame='J2000')

        indices = Pair([(0,0),(0,10),(20,0),(20,10),(20,11)])

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv, indices.to_pair((1,0)))

        (uv,time) = obs.uvt(indices, remask=True)

        self.assertEqual(uv[:4], indices.to_pair((1,0))[:4])
        self.assertTrue(np.all(uv.mask == 4*[False] + [True]))

        # Alternative axis order ('v', 'a', 'u')
        obs = Snapshot(('v','a','u'), tstart=98., texp=2.,
                       fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0,-1,0),(0,99,10),(20,-9,0),(20,77,10),(20,44,11)])
        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv, indices.to_pair((2,0)))

        (uv,time) = obs.uvt(indices, remask=True)

        self.assertEqual(uv[:4], indices.to_pair((2,0))[:4])
        self.assertTrue(np.all(uv.mask == 4*[False] + [True]))

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
