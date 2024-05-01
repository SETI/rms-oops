################################################################################
# oops/cadence/sequence.py: Sequence subclass of class Cadence
################################################################################

import unittest

import oops
from tests.cadence.test_metronome import (case_continuous, case_discontinuous,
                                          case_non_unique, case_partial_overlap)

class Test_Sequence(unittest.TestCase):

    def runTest(self):

        import numpy.random as random

        random.seed(5995)

        # These are the tests for subclass Metronome. We define Sequences so
        # that behavior should be identical, except in the out-of-bound cases

        ############################################
        # Tests for continuous case
        # 100-110, 110-120, 120-130, 130-140
        ############################################

        # cadence = oops.cadence.Metronome(100., 10., 10., 4)
        cadence = oops.cadence.Sequence([100.,110.,120.,130.,140.], 0.)
        case_continuous(self, cadence)

        ############################################
        # Discontinuous case, simulating the equivalent Metronome
        # 100-107.5, 110-117.5, 120-127.5, 130-137.5
        ############################################

        # cadence = oops.cadence.Metronome(100., 10., 7.5, 4)
        cadence = oops.cadence.Sequence([100.,110.,120.,130.], 7.5)
        case_discontinuous(self, cadence)

        ############################################
        # Non-unique case, simulating the equivalent Metronome
        # 100-140, 110-150, 120-160, 130-170
        ############################################

        # cadence = oops.cadence.Metronome(100., 10., 40., 4)
        cadence = oops.cadence.Sequence([100.,110.,120.,130.], 40.)
        case_non_unique(self, cadence)

        ############################################
        # Partial overlap case, simulating the equivalent Metronome
        # 100-140, 130-170, 160-200, 190-230
        ############################################

        # cadence = oops.cadence.Metronome(100., 30., 40., 4)
        cadence = oops.cadence.Sequence([100.,130.,160.,190.], 40.)
        case_partial_overlap(self, cadence)

        ############################################
        # Other cases
        ############################################

        cadence = oops.cadence.Sequence([100.,110.,120.,130.], [10.,10.,5.,10.])
        self.assertFalse(cadence.is_continuous)
        cadence = oops.cadence.Sequence([100.,110.,125.,130.], [10.,15.,5.,10.])
        self.assertTrue(cadence.is_continuous)

        self.assertEqual(cadence.tstep_at_time(105., remask=True), 0.5)
        self.assertEqual(cadence.tstep_at_time(115., remask=True), 4./3.)
        self.assertEqual(cadence.tstep_at_time(127., remask=True), 2.4)
        self.assertEqual(cadence.time_at_tstep(0.5 , remask=True), 105.)
        self.assertEqual(cadence.time_at_tstep(4./3., remask=True), 115.)
        self.assertEqual(cadence.time_at_tstep(2.4 , remask=True), 127.)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
