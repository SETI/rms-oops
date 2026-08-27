################################################################################
# tests/fov/test_offsetfov.py
################################################################################

import numpy as np
import pickle
import unittest

from polymath import Pair
from oops.fov import FlatFOV, OffsetFOV


class Test_OffsetFOV(unittest.TestCase):

    def runTest(self):

        np.random.seed(4712)

        flat = FlatFOV((1.e-4, 1.2e-4), (60, 40))
        UV_OFFSET = (1., 2.)

        # An offset given in any array-like form becomes a Pair, so that the
        # Fittable interface, which reads uv_offset.vals, always works
        for arg in (UV_OFFSET, list(UV_OFFSET), np.array(UV_OFFSET),
                    Pair(UV_OFFSET)):
            fov = OffsetFOV(flat, uv_offset=arg)
            self.assertIsInstance(fov.uv_offset, Pair)
            self.assertIsInstance(fov.xy_offset, Pair)
            self.assertEqual(fov.params, UV_OFFSET)
            self.assertEqual(fov.nparams, 2)

            # The fitting path accepts the same object
            fov.set_params((7., 8.))
            self.assertEqual(fov.params, (7., 8.))

        # The default is a zero offset
        self.assertEqual(OffsetFOV(flat).params, (0., 0.))

        # Only one of the two offsets may be given
        self.assertRaisesRegex(ValueError,
                               'only one of uv_offset and xy_offset',
                               OffsetFOV, flat, UV_OFFSET, UV_OFFSET)

        # An offset given in (x,y) describes the same FOV as the equivalent
        # offset given in (u,v)
        by_uv = OffsetFOV(flat, uv_offset=UV_OFFSET)
        by_xy = OffsetFOV(flat, xy_offset=by_uv.xy_offset)
        self.assertEqual(by_xy.params, UV_OFFSET)
        self.assertEqual(by_xy.uv_offset, by_uv.uv_offset)

        # The offset displaces (x,y) by exactly xy_offset
        uv = Pair(np.random.rand(200, 2) * np.array([60., 40.]))
        self.assertEqual(by_uv.xy_from_uvt(uv),
                         flat.xy_from_uvt(uv) - by_uv.xy_offset)

        # Pickling and copying both restore the offset. Only uv_offset is
        # saved, because the constructor derives xy_offset from it and refuses
        # to accept both.
        for fov in (by_uv, by_xy, OffsetFOV(flat)):
            for restored in (pickle.loads(pickle.dumps(fov)), fov.copy()):
                self.assertIsInstance(restored, OffsetFOV)
                self.assertEqual(restored.params, fov.params)
                self.assertEqual(restored.uv_offset, fov.uv_offset)
                self.assertEqual(restored.xy_offset, fov.xy_offset)
                self.assertEqual(restored.xy_from_uvt(uv), fov.xy_from_uvt(uv))

        # A copy is fittable, and fitting it leaves the original alone
        original = OffsetFOV(flat, uv_offset=UV_OFFSET)
        copied = original.copy()
        self.assertIsNot(copied, original)
        self.assertFalse(copied.is_frozen)
        copied.set_params((7., 8.))
        self.assertEqual(copied.params, (7., 8.))
        self.assertEqual(original.params, UV_OFFSET)

        # The underlying FOV is shared rather than duplicated
        self.assertIs(copied.fov, original.fov)

        # An unpickled object, by contrast, is frozen
        self.assertTrue(pickle.loads(pickle.dumps(original)).is_frozen)

################################################################################
