##########################################################################################
# tests/cadence/test_instant.py
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath import Pair, Scalar, Vector
import oops


def test_instant_defines_every_cadence_attribute() -> None:
    """An Instant fills in all eight attributes the Cadence contract requires."""

    cad = oops.cadence.Instant([100., 110., 130.])

    assert cad.shape == (3,)
    assert cad.time == (100., 130.)
    assert cad.midtime == 115.
    assert cad.lasttime == 130.
    assert cad.is_continuous is False
    assert cad.is_unique is True
    assert cad.min_tstride == 10.
    assert cad.max_tstride == 20.


def test_instant_is_not_unique_when_a_time_repeats() -> None:
    """Two time steps sharing one time make the cadence non-unique."""

    assert oops.cadence.Instant([100., 110., 100.]).is_unique is False


def test_instant_of_a_single_time_has_zero_strides() -> None:
    """A lone time step has no neighbor, so both strides are zero."""

    cad = oops.cadence.Instant(100.)

    assert cad.shape == ()
    assert cad.time == (100., 100.)
    assert cad.min_tstride == 0.
    assert cad.max_tstride == 0.
    assert cad.is_unique is True


def test_instant_ignores_masked_times() -> None:
    """Masked times take no part in the time limits or in the strides."""

    cad = oops.cadence.Instant(Scalar([100., 500., 110.], [False, True, False]))

    assert cad.time == (100., 110.)
    assert cad.max_tstride == 10.


def test_instant_requires_one_unmasked_time() -> None:
    """A fully masked input leaves no times to define the cadence."""

    with pytest.raises(ValueError, match='at least one unmasked time'):
        oops.cadence.Instant(Scalar([100., 110.], [True, True]))


def test_instant_survives_a_pickle_round_trip() -> None:
    """__getstate__ returns a tuple, so the cadence can be pickled."""

    cad = pickle.loads(pickle.dumps(oops.cadence.Instant([100., 110., 130.])))

    assert cad.time == (100., 130.)
    assert cad.max_tstride == 20.


def test_instant_time_at_tstep_selects_the_indexed_time() -> None:
    """Each time step reports its own time, not the whole table."""

    cad = oops.cadence.Instant([100., 110., 130.])

    assert cad.time_at_tstep(0) == 100.
    assert cad.time_at_tstep(1) == 110.
    assert cad.time_at_tstep(2) == 130.


def test_instant_time_at_tstep_takes_the_shape_of_its_index() -> None:
    """An array of time steps returns one time apiece."""

    cad = oops.cadence.Instant([100., 110., 130.])
    times = cad.time_at_tstep(Scalar([2, 0]))

    assert times.shape == (2,)
    assert times == Scalar([130., 100.])


def test_instant_time_at_tstep_truncates_a_fractional_index() -> None:
    """A time step is instantaneous, so a fractional index cannot be interpolated."""

    cad = oops.cadence.Instant([100., 110., 130.])

    assert cad.time_at_tstep(1.7) == 110.


def test_instant_time_at_tstep_clips_beyond_the_ends() -> None:
    """An index outside the cadence returns the time at the nearest edge."""

    cad = oops.cadence.Instant([100., 110., 130.])

    assert cad.time_at_tstep(-1) == 100.
    assert cad.time_at_tstep(5) == 130.


def test_instant_time_at_tstep_masks_outside_indices_on_request() -> None:
    """With remask, an index outside the cadence is masked rather than clipped."""

    cad = oops.cadence.Instant([100., 110., 130.])
    times = cad.time_at_tstep(Scalar([-1, 0, 2, 3, 4]), remask=True)

    assert list(times.mask) == [True, False, False, False, True]


def test_instant_time_at_tstep_masks_the_last_index_when_not_inclusive() -> None:
    """The index equal to the shape belongs to the cadence only when inclusive."""

    cad = oops.cadence.Instant([100., 110., 130.])

    assert cad.time_at_tstep(3, remask=True, inclusive=True) == 130.
    assert cad.time_at_tstep(3, remask=True, inclusive=False).mask


def test_instant_time_range_at_tstep_has_zero_duration() -> None:
    """An Instant has no duration, so a time step starts and ends at its own time."""

    cad = oops.cadence.Instant([100., 110., 130.])
    (time0, time1) = cad.time_range_at_tstep(1)

    assert time0 == 110.
    assert time1 == 110.


def test_instant_tstep_at_time_finds_the_matching_step() -> None:
    """A sampled time reports the index of its own time step."""

    cad = oops.cadence.Instant([100., 110., 130.])

    assert cad.tstep_at_time(100.) == 0
    assert cad.tstep_at_time(110.) == 1
    assert cad.tstep_at_time(130.) == 2


def test_instant_tstep_at_time_masks_an_unsampled_time() -> None:
    """An Instant samples isolated moments, so any other time has no time step."""

    cad = oops.cadence.Instant([100., 110., 130.])

    assert cad.tstep_at_time(105.).mask
    assert cad.tstep_at_time(1.e6).mask


def test_instant_tstep_at_time_takes_the_shape_of_its_time() -> None:
    """The result is shaped by the given times, not by the shape of the cadence."""

    cad = oops.cadence.Instant([100., 110., 130.])
    tsteps = cad.tstep_at_time(Scalar([130., 100.]))

    assert tsteps.shape == (2,)
    assert tsteps == Scalar([2, 0])


def test_instant_tstep_at_time_returns_the_first_of_repeated_times() -> None:
    """Where one time is tabulated twice, the earlier time step is the one reported."""

    cad = oops.cadence.Instant([100., 110., 100.])

    assert cad.is_unique is False
    assert cad.tstep_at_time(100.) == 0


def test_instant_ignores_masked_steps_in_both_directions() -> None:
    """A masked time step supplies no time and matches no time."""

    cad = oops.cadence.Instant(Scalar([100., 500., 110.], [False, True, False]))

    assert cad.time_at_tstep(1).mask                # the masked step has no time
    assert cad.tstep_at_time(500.).mask             # and cannot be found by its time
    assert cad.tstep_at_time(110.) == 2


def test_instant_tstep_range_at_time_spans_one_step() -> None:
    """A sampled time is active in exactly one time step."""

    cad = oops.cadence.Instant([100., 110., 130.])
    (first, last) = cad.tstep_range_at_time(110.)

    assert first == 1
    assert last == 2


def test_instant_tstep_range_at_time_is_empty_when_unsampled() -> None:
    """A time the cadence does not sample is active in no time step at all."""

    cad = oops.cadence.Instant([100., 110., 130.])
    (first, last) = cad.tstep_range_at_time(105.)

    assert first == last


def test_instant_time_is_outside_only_at_the_sampled_times() -> None:
    """Every time other than a tabulated one falls outside the cadence."""

    cad = oops.cadence.Instant([100., 110., 130.])
    outside = cad.time_is_outside(Scalar([100., 105., 130., 1.e6]))

    assert outside.shape == (4,)
    assert list(outside.vals) == [False, True, False, True]


def test_instant_indexes_a_two_dimensional_table() -> None:
    """A 2-D Instant is indexed by a Pair, one component per axis."""

    cad = oops.cadence.Instant([[100., 110.], [120., 130.]])

    assert cad.shape == (2, 2)
    assert cad.time_at_tstep(Pair((1, 0))) == 120.
    assert cad.time_at_tstep(Pair((0, 1))) == 110.
    assert cad.tstep_at_time(120.) == Pair((1, 0))


def test_instant_round_trips_every_two_dimensional_step() -> None:
    """Converting a time step to its time and back returns the same time step."""

    cad = oops.cadence.Instant([[100., 110.], [120., 130.]])

    for tstep in (Pair((0, 0)), Pair((0, 1)), Pair((1, 0)), Pair((1, 1))):
        assert cad.tstep_at_time(cad.time_at_tstep(tstep)) == tstep


def test_instant_of_a_single_time_has_one_time_step() -> None:
    """A cadence of shape () holds one time, which its lone time step reports."""

    cad = oops.cadence.Instant(100.)

    assert cad.time_at_tstep(0) == 100.
    assert cad.time_range_at_tstep(0) == (100., 100.)
    assert cad.tstep_at_time(100.) == 0
    assert cad.tstep_at_time(101.).mask
    assert cad.time_is_outside(100.) == False       # noqa: E712  numpy bool, not the
                                                    # False singleton

def test_a_multidimensional_instant_reports_a_pair_of_step_ranges() -> None:
    """With a 2-D array of moments, a step range is a pair of indices."""

    cad = oops.cadence.Instant([[100., 110.], [120., 130.]])

    (first, last) = cad.tstep_range_at_time(Scalar([110., 105.]))

    assert isinstance(first, Pair)
    assert first == Pair([(0, 1), (0, 0)])
    assert last == Pair([(1, 2), (0, 0)])       # the second time is not sampled


def test_a_three_dimensional_instant_reports_a_vector_of_step_ranges() -> None:
    """Beyond two dimensions the indices are returned as a Vector."""

    cad = oops.cadence.Instant(np.arange(8.).reshape(2, 2, 2))

    (first, last) = cad.tstep_range_at_time(Scalar(5.))

    assert isinstance(first, Vector)
    assert first == Vector((1, 0, 1))
    assert last == Vector((2, 1, 2))    # the upper limit is exclusive on every axis


def test_a_shapeless_instant_reports_a_shapeless_step_range() -> None:
    """One moment alone has a single time step, index zero."""

    cad = oops.cadence.Instant(100.)

    (first, last) = cad.tstep_range_at_time(Scalar([100., 101.]))

    assert first == Scalar([0, 0])
    assert last == Scalar([1, 0])               # the second time is not sampled


def test_shifting_an_instant_moves_every_moment() -> None:
    """A shifted Instant samples the same moments, later by the offset."""

    cad = oops.cadence.Instant([100., 110., 130.])

    shifted = cad.time_shift(50.)

    assert shifted.time == (150., 180.)
    assert shifted.time_at_tstep(Scalar(1)) == Scalar(160.)


def test_an_instant_can_be_declared_continuous() -> None:
    """as_continuous returns a copy that claims continuity, leaving the original alone."""

    cad = oops.cadence.Instant([100., 110., 130.])

    continuous = cad.as_continuous()

    assert continuous.is_continuous
    assert not cad.is_continuous
    assert continuous.time == cad.time

##########################################################################################
