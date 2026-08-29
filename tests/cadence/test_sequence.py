##########################################################################################
# oops/cadence/sequence.py: Sequence subclass of class Cadence
##########################################################################################

import oops
from tests.cadence.test_metronome import (case_continuous, case_discontinuous,
                                          case_non_unique, case_partial_overlap)

def test_sequence():
    import numpy.random as random

    random.seed(5995)

    # These are the tests for subclass Metronome. We define Sequences so
    # that behavior should be identical, except in the out-of-bound cases

    ######################################################################################
    # Tests for continuous case
    # 100-110, 110-120, 120-130, 130-140
    ######################################################################################

    # cadence = oops.cadence.Metronome(100., 10., 10., 4)
    cadence = oops.cadence.Sequence([100.,110.,120.,130.,140.], 0.)
    case_continuous(cadence)

    ######################################################################################
    # Discontinuous case, simulating the equivalent Metronome
    # 100-107.5, 110-117.5, 120-127.5, 130-137.5
    ######################################################################################

    # cadence = oops.cadence.Metronome(100., 10., 7.5, 4)
    cadence = oops.cadence.Sequence([100.,110.,120.,130.], 7.5)
    case_discontinuous(cadence)

    ######################################################################################
    # Non-unique case, simulating the equivalent Metronome
    # 100-140, 110-150, 120-160, 130-170
    ######################################################################################

    # cadence = oops.cadence.Metronome(100., 10., 40., 4)
    cadence = oops.cadence.Sequence([100.,110.,120.,130.], 40.)
    case_non_unique(cadence)

    ######################################################################################
    # Partial overlap case, simulating the equivalent Metronome
    # 100-140, 130-170, 160-200, 190-230
    ######################################################################################

    # cadence = oops.cadence.Metronome(100., 30., 40., 4)
    cadence = oops.cadence.Sequence([100.,130.,160.,190.], 40.)
    case_partial_overlap(cadence)

    ######################################################################################
    # Other cases
    ######################################################################################

    cadence = oops.cadence.Sequence([100.,110.,120.,130.], [10.,10.,5.,10.])
    assert not cadence.is_continuous
    cadence = oops.cadence.Sequence([100.,110.,125.,130.], [10.,15.,5.,10.])
    assert cadence.is_continuous

    assert cadence.tstep_at_time(105., remask=True) == 0.5
    assert cadence.tstep_at_time(115., remask=True) == 4./3.
    assert cadence.tstep_at_time(127., remask=True) == 2.4
    assert cadence.time_at_tstep(0.5 , remask=True) == 105.
    assert cadence.time_at_tstep(4./3., remask=True) == 115.
    assert cadence.time_at_tstep(2.4 , remask=True) == 127.
##########################################################################################
