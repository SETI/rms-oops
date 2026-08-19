################################################################################
# oops/cadence/timeshift.py: TimeShift subclass of class Cadence
################################################################################

import numpy as np
import unittest

from polymath import Scalar
from oops.cadence import Metronome, TimeShift


class Test_TimeShift(unittest.TestCase):

    def runTest(self):

        DT = 10.

        # 100-110, 110-120, 120-130, 130-140
        cadence = Metronome(100., 10., 10., 4)
        shifted = TimeShift(DT, cadence)
        self.assertEqual(shifted.dt, DT)
        self.assertIsNone(shifted.link)

        # Every time is offset by dt, and the shape is unchanged
        self.assertEqual(shifted.shape, cadence.shape)
        self.assertEqual(shifted.time, (cadence.time[0] + DT, cadence.time[1] + DT))
        self.assertEqual(shifted.midtime, cadence.midtime + DT)

        tstep = Scalar(np.arange(0., 4., 0.25))
        self.assertEqual(shifted.time_at_tstep(tstep),
                         cadence.time_at_tstep(tstep) + DT)
        self.assertEqual(shifted.tstep_at_time(cadence.time_at_tstep(tstep) + DT), tstep)

        # A linked TimeShift tracks the offset of the object it is linked to
        linked = TimeShift(shifted, cadence)
        self.assertIs(linked.link, shifted)
        self.assertEqual(linked.dt, DT)

        shifted.set_params(np.array([2. * DT]))
        linked._refresh()
        self.assertEqual(linked.dt, 2. * DT)
        self.assertEqual(linked.time_at_tstep(tstep),
                         cadence.time_at_tstep(tstep) + 2. * DT)

################################################################################
