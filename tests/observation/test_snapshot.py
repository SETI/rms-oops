################################################################################
# tests/observation/test_snapshot.py
################################################################################

import numpy as np
import unittest

from polymath         import Matrix3, Pair, Scalar, Vector
from oops.fov         import FlatFOV
from oops.frame       import Cmatrix, SpinFrame
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

        # cmatrix()
        m = Matrix3([[0,1,0],[0,0,-1],[-1,0,0]])
        cmatrix_frame = Cmatrix(m, frame_id='TEST_SNAPSHOT_CMATRIX')

        obs = Snapshot(('u','v'), tstart=98., texp=2.,
                       fov=fov, path='SSB', frame=cmatrix_frame)

        self.assertTrue(np.all(obs.cmatrix().vals == m.vals))
        self.assertTrue(np.all(obs.cmatrix(uv=(3,4)).vals == m.vals))
        self.assertTrue(np.all(obs.cmatrix(time=99.).vals == m.vals))
        self.assertTrue(np.all(obs.cmatrix(reference=obs.frame).vals
                               == np.eye(3)))

        # cmatrix() with a time-dependent frame, using both uv and time
        #
        # The Cmatrix frame above is fixed in time, so it cannot reveal whether
        # cmatrix() actually consults uv and time. Here we combine a rotating
        # SpinFrame (a different matrix at every time) with an observation whose
        # midtime_at_uv() varies with the u coordinate, so distinct UVs must map
        # to distinct times and therefore distinct matrices.

        spin = SpinFrame(0., 1., 0., 2, 'J2000',
                         frame_id='TEST_SNAPSHOT_SPIN')

        def spin_matrix(t):
            # The J2000 -> SpinFrame rotation at time t (see SpinFrame, axis=2)
            (c, s) = (np.cos(t), np.sin(t))
            return np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])

        class TimeDependentSnapshot(Snapshot):
            # midtime depends on the u coordinate: midtime == u
            def midtime_at_uv(self, uv, tfrac=0.5):
                return Scalar.as_scalar(Pair.as_pair(uv).to_scalar(0))

        obs = TimeDependentSnapshot(('u','v'), tstart=98., texp=2.,
                                    fov=fov, path='SSB', frame=spin)

        # Selecting different UVs selects different times -> different matrices
        cmat1 = obs.cmatrix(uv=(1.,4.))
        cmat2 = obs.cmatrix(uv=(2.,4.))
        self.assertTrue(np.allclose(cmat1.vals, spin_matrix(1.)))
        self.assertTrue(np.allclose(cmat2.vals, spin_matrix(2.)))
        self.assertFalse(np.allclose(cmat1.vals, cmat2.vals))

        # The default uv is the center of the FOV (uv_shape/2 -> u == 5)
        self.assertTrue(np.allclose(obs.cmatrix().vals, spin_matrix(5.)))

        # An explicit time is used directly, overriding uv
        self.assertTrue(np.allclose(obs.cmatrix(time=3.).vals, spin_matrix(3.)))
        self.assertTrue(np.allclose(obs.cmatrix(uv=(1.,4.), time=3.).vals,
                                    spin_matrix(3.)))

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
