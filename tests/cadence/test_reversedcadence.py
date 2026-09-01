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


def test_tstep_at_time_inverts_time_at_tstep():
    """Every index recovers itself through the time it maps to.

    Only whole steps are reversed; within a step, time still increases with the index, so
    the fractional part of the index must survive the round trip unchanged. Reversing the
    index outright would invert that fraction, which agrees with the correct value only at
    the midpoint of each step.
    """

    for inner in [oops.cadence.Metronome(100., 10., 10., 4),    # continuous
                  oops.cadence.Metronome(100., 10., 7.5, 4),    # gapped
                  oops.cadence.Sequence([100., 110., 130., 160.], 10.)]:
        cadence = oops.cadence.ReversedCadence(inner)

        # Sample away from the step boundaries, where a time belongs to two indices
        tstep = np.arange(0.05, cadence.shape[0], 0.1)
        time = cadence.time_at_tstep(tstep)

        assert abs(cadence.tstep_at_time(time) - tstep).max() < 1.e-12


def test_tstep_at_time_increases_with_time_inside_a_step():
    """The index rises with time within a step, because only whole steps are reversed."""

    cadence = oops.cadence.ReversedCadence(oops.cadence.Metronome(100., 10., 10., 4))

    time = oops.Scalar(135.)
    time.insert_deriv('t', oops.Scalar(1.))

    assert cadence.tstep_at_time(time, derivs=True).d_dt == 0.1


def test_double_reversal_restores_the_original_cadence():
    """Reversing a cadence twice restores it, except at the very end of its time range.

    A time at the end of a step maps to the exclusive upper bound of that step, which is
    the index where the next step begins. One reversal sends the end of the underlying
    cadence to the end of this cadence's first step, and a second reversal can no longer
    tell that index apart from the start of its second step. Every time strictly inside
    the range is unaffected, as is `time_at_tstep` throughout.
    """

    for args in [(100., 10., 10., 4),      # continuous
                 (100., 10., 7.5, 4),      # discontinuous
                 (100., 10., 40., 4),      # non-unique
                 (100., 30., 40., 4)]:     # partially overlapping
        cadence = oops.cadence.Metronome(*args)
        doubled = oops.cadence.ReversedCadence(oops.cadence.ReversedCadence(cadence))

        assert doubled.shape == cadence.shape
        assert doubled.time == cadence.time
        assert doubled.midtime == cadence.midtime
        assert doubled.lasttime == cadence.lasttime
        assert doubled.is_continuous == cadence.is_continuous
        assert doubled.is_unique == cadence.is_unique

        # Indices map to the same times, including the shifted end of the last step
        for tstep in np.arange(0., cadence.shape[0] + 0.001, 0.125):
            assert doubled.time_at_tstep(tstep) == cadence.time_at_tstep(tstep)
            assert (doubled.time_range_at_tstep(tstep)
                    == cadence.time_range_at_tstep(tstep))

        # Times map to the same indices everywhere strictly inside the time range
        (start, stop) = cadence.time
        for time in np.arange(start - 5., stop, 0.5):
            assert abs(doubled.tstep_at_time(time)
                       - cadence.tstep_at_time(time)) < 1.e-12
            assert (doubled.tstep_at_time(time, remask=True).mask
                    == cadence.tstep_at_time(time, remask=True).mask)
            assert doubled.tstep_range_at_time(time) == cadence.tstep_range_at_time(time)

        # The documented exception: at the end, the original reports the index one past
        # its last step, while the doubled cadence reports where that time falls within
        # the step it belongs to.
        assert cadence.tstep_at_time(stop) == cadence.shape[0]
        assert abs(doubled.tstep_at_time(stop) - 2.) < 1.e-12


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
