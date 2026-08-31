##########################################################################################
# tests/cadence/test_reversedcadence.py
##########################################################################################

import pickle

import numpy as np
import pytest

import oops
from tests.cadence.test_tdicadence import (case_tdicadence_10_100_10_2_down,
                                           case_tdicadence_10_100_10_2_up,
                                           case_tdicadence_100_1000_10_100_down)
from tests.cadence.test_metronome import (case_continuous, case_discontinuous,
                                          case_non_unique, case_partial_overlap)


# Test using TDICadence, which already has the feature that the "up" and the
# "down" versions are index-reversed.
def test_reversedcadence():
    np.random.seed(3547)

    ######################################################################################
    # 10 lines, 2 stages, TDI downward, 100-120
    ######################################################################################

    tdicad = oops.cadence.TDICadence(10, 100., 10., 2, tdi_sign=1)
    cad = oops.cadence.ReversedCadence(tdicad)
    case_tdicadence_10_100_10_2_down(cad)

    ######################################################################################
    # 10 lines, 2 stages, TDI upward
    ######################################################################################

    tdicad = oops.cadence.TDICadence(10, 100., 10., 2, tdi_sign=-1)
    cad = oops.cadence.ReversedCadence(tdicad)
    case_tdicadence_10_100_10_2_up(cad)

    ######################################################################################
    # 100 lines, 100 stages, TDI downward
    ######################################################################################

    tdicad = oops.cadence.TDICadence(100, 1000., 10., 100, tdi_sign=1)
    cad = oops.cadence.ReversedCadence(tdicad)
    case_tdicadence_100_1000_10_100_down(cad)

    ######################################################################################
    # Doubly-reversed Metronome, continuous
    # 100-110, 110-120, 120-130, 130-140
    ######################################################################################

    cadence = oops.cadence.Metronome(100., 10., 10., 4)
    cadence = oops.cadence.ReversedCadence(oops.cadence.ReversedCadence(cadence))
    case_continuous(cadence)

    ######################################################################################
    # Doubly-reversed Metronome, discontinuous case
    # 100-107.5, 110-117.5, 120-127.5, 130-137.5
    ######################################################################################

    cadence = oops.cadence.Metronome(100., 10., 7.5, 4)
    cadence = oops.cadence.ReversedCadence(oops.cadence.ReversedCadence(cadence))
    case_discontinuous(cadence)

    ######################################################################################
    # Doubly-reversed Metronome, non-unique case
    # 100-140, 110-150, 120-160, 130-170
    ######################################################################################

    cadence = oops.cadence.Metronome(100., 10., 40., 4)
    cadence = oops.cadence.ReversedCadence(oops.cadence.ReversedCadence(cadence))
    case_non_unique(cadence)

    ######################################################################################
    # Doubly-reversed Metronome, partial overlap case
    # 100-140, 130-170, 160-200, 190-230
    ######################################################################################

    cadence = oops.cadence.Metronome(100., 30., 40., 4)
    cadence = oops.cadence.ReversedCadence(oops.cadence.ReversedCadence(cadence))
    case_partial_overlap(cadence)


def test_reversedcadence_requires_axis_zero() -> None:
    """Only axis 0 can be reversed, and the axis carries into every derived cadence."""

    cadence = oops.cadence.Metronome(100., 10., 8., 4)

    with pytest.raises(ValueError, match='axis must be 0'):
        oops.cadence.ReversedCadence(cadence, 1)

    reversed_ = oops.cadence.ReversedCadence(cadence, 0)

    assert reversed_._axis == 0
    assert reversed_.time_shift(5.)._axis == 0
    assert reversed_.as_continuous()._axis == 0
    assert pickle.loads(pickle.dumps(reversed_))._axis == 0

##########################################################################################
