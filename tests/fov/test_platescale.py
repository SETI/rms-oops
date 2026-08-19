################################################################################
# oops/fov/platescale.py: Platescale subclass of class FOV
################################################################################

import numpy as np
import unittest

from polymath import Pair
from oops.fov import FlatFOV, Platescale


class Test_Platescale(unittest.TestCase):

    def runTest(self):

        np.random.seed(1170)

        FACTOR = 2.
        flat = FlatFOV((1.e-4, 1.2e-4), (60, 40))
        scaled = Platescale(FACTOR, flat)

        self.assertEqual(scaled.nparams, 1)
        self.assertEqual(scaled.params, (FACTOR,))

        # The pixels cover the same range of (u,v), but each subtends more sky
        self.assertEqual(scaled.uv_shape, flat.uv_shape)
        self.assertEqual(scaled.uv_los, flat.uv_los)
        self.assertEqual(scaled.uv_scale, flat.uv_scale * FACTOR)

        # (x,y) scale with the factor, so the area of a pixel scales with its square
        self.assertEqual(scaled.uv_area, flat.uv_area * FACTOR**2)

        uv = Pair(np.random.rand(200, 2) * np.array([60., 40.]))
        self.assertEqual(scaled.xy_from_uvt(uv), flat.xy_from_uvt(uv) * FACTOR)

        # ...which keeps the area of a pixel at its nominal value. Were uv_area not
        # scaled, this factor would come out at the square of the plate scale instead
        self.assertTrue(np.all(np.abs(scaled.area_factor(uv).vals - 1.) < 1.e-3))
        self.assertTrue(np.all(np.abs(flat.area_factor(uv).vals - 1.) < 1.e-3))

        # (u,v) and (x,y) still convert back and forth
        self.assertTrue(np.all(np.abs(scaled.uv_from_xyt(scaled.xy_from_uvt(uv)).vals
                                      - uv.vals) < 1.e-12))

        # A factor of one leaves the FOV alone
        unscaled = Platescale(1., flat)
        self.assertEqual(unscaled.uv_scale, flat.uv_scale)
        self.assertEqual(unscaled.uv_area, flat.uv_area)
        self.assertEqual(unscaled.xy_from_uvt(uv), flat.xy_from_uvt(uv))

        # Refitting the scale factor updates everything that depends on it
        scaled.set_params(np.array([3.]))
        self.assertEqual(scaled.factor, 3.)
        self.assertEqual(scaled.params, (3.,))
        self.assertEqual(scaled.uv_scale, flat.uv_scale * 3.)
        self.assertEqual(scaled.uv_area, flat.uv_area * 9.)
        self.assertEqual(scaled.xy_from_uvt(uv), flat.xy_from_uvt(uv) * 3.)

        # The wrong number of parameters is rejected
        self.assertRaisesRegex(ValueError, 'requires 1 fit',
                               scaled.set_params, np.array([1., 2.]))

        # Freezing prevents any further refitting
        scaled.freeze()
        self.assertTrue(scaled.is_frozen)
        self.assertRaisesRegex(ValueError, 'frozen',
                               scaled.set_params, np.array([4.]))

################################################################################
