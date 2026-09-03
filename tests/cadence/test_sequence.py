##########################################################################################
# oops/cadence/sequence.py: Sequence subclass of class Cadence
##########################################################################################

import pickle

import pytest

from polymath import Scalar
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
# Constructor validation, serialization, and the shortcuts of the Cadence API
##########################################################################################

TLIST = [100., 110., 120., 130.]


def test_a_masked_time_list_is_rejected() -> None:
    """Every time in the list has to be a real time."""

    tlist = Scalar(TLIST, [False, True, False, False])

    with pytest.raises(ValueError, match='tlist input must be unmasked'):
        oops.cadence.Sequence(tlist, 10.)


def test_a_masked_exposure_list_is_rejected() -> None:
    """Every exposure time has to be a real duration."""

    texp = Scalar([10., 10., 10., 10.], [False, False, True, False])

    with pytest.raises(ValueError, match='texp input must be unmasked'):
        oops.cadence.Sequence(TLIST, texp)


@pytest.mark.parametrize('tlist', [[[100., 110.], [120., 130.]], [100.]],
                         ids=['two-dimensional', 'one-value'])
def test_the_time_list_must_be_one_dimensional_with_more_than_one_value(tlist) -> None:
    """A Sequence runs along one axis and needs at least two times to define a stride."""

    with pytest.raises(ValueError, match='tlist must be 1-D'):
        oops.cadence.Sequence(tlist, 10.)


def test_an_exposure_list_of_the_wrong_length_is_rejected() -> None:
    """One exposure time per time step, and no more."""

    with pytest.raises(ValueError, match='Shape mismatch between texp and tlist'):
        oops.cadence.Sequence(TLIST, [10., 10., 10.])


def test_a_non_positive_exposure_in_a_list_is_rejected() -> None:
    """A time step of zero or negative duration samples nothing."""

    with pytest.raises(ValueError, match='All texp values must be positive'):
        oops.cadence.Sequence(TLIST, [10., 10., 0., 10.])


def test_a_non_positive_constant_exposure_is_rejected() -> None:
    """The same applies when one exposure time is given for every step."""

    with pytest.raises(ValueError, match='All texp values must be positive'):
        oops.cadence.Sequence(TLIST, -10.)


def test_a_time_list_that_is_not_monotonic_is_rejected() -> None:
    """With texp derived from the strides, the times must increase."""

    with pytest.raises(ValueError, match='tlist inputs must be monotonic'):
        oops.cadence.Sequence([100., 120., 110., 130.], 0.)


def test_a_sequence_survives_a_round_trip_through_pickle() -> None:
    """Unpickling rebuilds the sequence from its times and exposures."""

    cad = oops.cadence.Sequence(TLIST, [8., 9., 10., 11.])

    revived = pickle.loads(pickle.dumps(cad))

    assert revived.time == cad.time
    assert revived.shape == cad.shape
    assert revived.time_at_tstep(Scalar(1.5)) == cad.time_at_tstep(Scalar(1.5))


def test_a_step_range_needs_ordered_stop_times() -> None:
    """Steps that end out of order cannot be searched by their stop times."""

    cad = oops.cadence.Sequence(TLIST, [100., 10., 10., 10.])
    assert not cad._tstop_is_ordered

    with pytest.raises(RuntimeError, match='stop times are not strictly ordered'):
        cad.tstep_range_at_time(Scalar(105.))


def test_a_shapeless_time_is_tested_against_its_own_step() -> None:
    """A single time is converted to one integer step index, not an array of them."""

    cad = oops.cadence.Sequence(TLIST, 8.)      # gaps between the steps

    assert not cad.time_is_outside(Scalar(105.))
    assert cad.time_is_outside(Scalar(109.))


def test_a_continuous_sequence_is_returned_unchanged() -> None:
    """A cadence with no gaps is already continuous, so it is its own continuous form."""

    cad = oops.cadence.Sequence(TLIST, 10.)
    assert cad.is_continuous

    assert cad.as_continuous() is cad

##########################################################################################
