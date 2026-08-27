################################################################################
# oops/path/quickpath.py: Subclass QuickPath of class Path
################################################################################

import numpy as np
import unittest

import cspyce

from polymath     import Scalar, Vector3
from oops.body    import Body
from oops.config  import QUICK
from oops.frame   import Frame
from oops.gravity import Gravity
from oops.path    import FixedPath, KeplerPath, Path, QuickPath, SpicePath
from oops.unittester_support import TEST_SPICE_PREFIX


class Test_QuickPath(unittest.TestCase):

    def setUp(self):
        paths = TEST_SPICE_PREFIX.retrieve(['naif0009.tls',
                                            'pck00010.tpc',
                                            'de421.bsp'])
        for path in paths:
            cspyce.furnsh(path)
        Path._reset_caches()
        Frame._reset_caches()

    def tearDown(self):
        Body.reset_registry()
        Path._reset_caches()
        Frame._reset_caches()

    def runTest(self):

        np.random.seed(9033)

        mars = SpicePath('MARS', 'SSB')
        epoch = 1.e8
        time = Scalar(epoch + np.arange(0., 100., 0.01))

        ########################################
        # Tabulating a Path does not spawn a second, nested QuickPath
        ########################################

        # SpicePath quickens itself when handed an array of times, so the tabulation
        # inside QuickPath must not re-enter that machinery. The span is short enough
        # that a QuickPath of the tabulation times would otherwise be judged worthwhile.
        short_time = Scalar(epoch + np.arange(0., 0.5, 0.001))
        self.assertIsInstance(mars.quick_path(short_time, quick={}), QuickPath)
        self.assertEqual(len(mars._quickpaths), 1)

        ########################################
        # A Path whose state is fixed in time is never tabulated
        ########################################

        # FixedPath returns one position and velocity regardless of the times requested,
        # so a QuickPath could not interpolate it and would gain nothing if it could
        fixed = FixedPath(Vector3((1.e3, 0., 0.)), mars, path_id='fixed_path')
        self.assertFalse(fixed._USE_QUICKPATHS)
        self.assertIs(fixed.quick_path(time, quick={}), fixed)

        self.assertRaises(ValueError, QuickPath, fixed, epoch, epoch + 100.,
                          QUICK.dictionary)

        ########################################
        # A composite Path inherits the opt-in of the paths it combines
        ########################################

        # The fixed offset above contributes nothing, but the SpicePath it is measured
        # from does, so the composite is worth tabulating
        linked = fixed.wrt(Path.SSB, Frame.J2000)
        self.assertTrue(linked._USE_QUICKPATHS)
        quick = linked.quick_path(time, quick={})
        self.assertIsInstance(quick, QuickPath)

        exact = linked.event_at_time(time, quick=False)
        interpolated = quick.event_at_time(time, quick=False)
        self.assertLess(np.max(np.abs((interpolated.pos - exact.pos).norm().vals)), 1.e-6)

        ########################################
        # A tabulation of a fittable Path is redone after that Path is re-fit
        ########################################

        Body._undefine_solar_system()
        Body.define_solar_system('2000-01-01', '2010-01-01')

        a = 140000.
        saturn = Gravity.lookup('SATURN')
        elements = [a, 1., saturn.n(a), 0.2, 3., saturn.dperi_dt(a),
                    0.1, 5., saturn.dnode_dt(a)]
        kepler = KeplerPath(Body.lookup('SATURN'), 0., elements, path_id='fitted_kepler')
        self.assertTrue(kepler._USE_QUICKPATHS)

        quick = kepler.quick_path(time, quick={})
        self.assertIsInstance(quick, QuickPath)

        # Enlarge the orbit by 10,000 km, which moves the body much farther than any
        # interpolation error
        elements[0] = a + 10000.
        kepler.set_params(np.array(elements))
        exact = kepler.event_at_time(time, quick=False)

        # The same QuickPath is handed back, but tabulated afresh
        reused = kepler.quick_path(time, quick={})
        self.assertIs(reused, quick)
        error = (reused.event_at_time(time, quick=False).pos - exact.pos).norm()
        self.assertLess(np.max(error.vals), 1.e-3)

################################################################################
